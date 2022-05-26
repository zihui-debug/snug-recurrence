import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import GRU,Dense
from tensorflow.keras import Sequential
import numpy as np
import math
import smpl
import snug_utils as utils
import argparse
from outfit import Outfit
import Data
import time
import gc
import psutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    print(r'当前显卡显存使用情况：总共 {} MB， 已经使用 {} MB， 剩余 {} MB'
          .format(total, used, free))
    return total, used, free

parser = argparse.ArgumentParser()

parser.add_argument(
    "--motion",
    type=str,
    default="assets/CMU/07/07_02_poses.npz",
    help="path of the motion to use as input"
)

parser.add_argument(
    "--garment",
    type=str,
    default="tshirt",
    help="name of the garment (tshirt, tank, top, pants or shorts)"
)

parser.add_argument(
    "--savedir",
    type=str,
    default="tmp",
    help="path to save the result"
)

args = parser.parse_args()

# Load smpl
body = smpl.SMPL("assets/SMPL/basicModel_f_lbs_10_207_0_v1.0.0.pkl")
v_Tbody = body.template_vertices
# Load garment model
model_path, template_path = utils.get_model_path(args.garment)
# snug = tf.saved_model.load(model_path)
v_Tgarment, f_garment = utils.load_obj(template_path)
outfit = Outfit(v_Tgarment, f_garment)
# Load motion
# poses, trans, trans_vel = utils.load_motion(args.motion)
# Body shape
betas = np.zeros(10, dtype=np.float32)

def pairwise_distance(A, B):
    rA = np.sum(np.square(A), axis=1)
    rB = np.sum(np.square(B), axis=1)
    distances = - 2*np.matmul(A, np.transpose(B)) + rA[:, np.newaxis] + rB[np.newaxis, :]
    return distances

def euclidean_dist(x, y, batch = False):
    square_x = tf.reduce_sum(tf.square(x), axis=-1)
    square_y = tf.reduce_sum(tf.square(y), axis=-1)
    # expand dims for broadcasting
    ex = tf.expand_dims(square_x, axis=-1)
    ey = tf.expand_dims(square_y, axis=-2)
    # XY matrix
    if batch:
        xy = tf.einsum('bij,bkj->bik', x, y)
    else:
        xy = tf.einsum('ij,kj->ik', x, y)
    # 如果没有batch_size这个维度，可以写成：
    # xy = tf.einsum('ij,kj->ik', x, y)
    # compute distance，浮点防溢出
    dist = tf.sqrt(ex - 2 * xy + ey + 1e-10)
    return dist

def find_nearest_neighbour(A, B, dtype=tf.int32):
    nearest_neighbour = tf.argmin(euclidean_dist(A, B), axis=-1)
    # return nearest_neighbour.astype(dtype)
    return nearest_neighbour

closest_vertices = find_nearest_neighbour(v_Tgarment, v_Tbody)

#计算服装蒙皮权重
# garment_skinning_weights = np.zeros((v_Tgarment.shape[0],24))
garment_skinning_weights = tf.gather(body.skinning_weights, indices=closest_vertices, axis=0)
# for i in range(0, v_Tgarment.shape[0]):
#     min_dis = 100000000
#     neighbor = -1
#     for j in range(0, tf.shape(v_Tbody)[0]):
#         vec = v_Tgarment[i]-v_Tbody[j]
#         dis = np.linalg.norm(vec)
#         if dis<min_dis:
#             min_dis=dis
#             neighbor=j
#     garment_skinning_weights[i]=body.skinning_weights[neighbor]
# print(garment_skinning_weights.shape)
# np.save('tshirt_skinning_weights',garment_skinning_weights)

class Deformation_model(tf.keras.Model):
    def __init__(self, output):
        super(Deformation_model, self).__init__()
        self.v_TGRAMENT=v_Tgarment
        self.g_skinweight = garment_skinning_weights
        self.gru1 = GRU(256,return_sequences=True,activation='tanh')
        self.gru2 = GRU(256, return_sequences=True, activation='tanh')
        self.gru3 = GRU(256, return_sequences=True, activation='tanh')
        self.gru4 = GRU(256, return_sequences=True, activation='tanh')

        self.linear = Dense(output)
    def call(self, inputs):
        # pose,translation_vel,shape,hidden_states0,hidden_states1,hidden_states2,hidden_states3,\
        # translation,shape_blendshape,joint_transforms,vertices,vertex_normals = inputs
        shape, pose, translation, trans_vel, hidden_states0,hidden_states1,hidden_states2,hidden_states3 = inputs
        shape = tf.reshape(tf.tile(shape,multiples=[pose.shape[0],pose.shape[1],1]),[pose.shape[0],pose.shape[1],10])
        x = tf.concat([shape,pose],axis=-1)
        x = self.gru1(x, initial_state=hidden_states0)
        x = self.gru2(x, initial_state=hidden_states1)
        x = self.gru3(x, initial_state=hidden_states2)
        x = self.gru4(x, initial_state=hidden_states3)
        x = self.linear(x)
        verts_num = v_Tgarment.shape[0]
        x = tf.reshape(x,[-1,pose.shape[1],verts_num,3])
        x0 = x[:,0,:]
        x1 = x[:,1,:]
        x2 = x[:,2,:]

        pose0 = pose[:,0,:]
        pose1 = pose[:,1,:]
        pose2 = pose[:,2,:]
        translation0 = translation[:,0,:]
        translation1 = translation[:,1,:]
        translation2 = translation[:,2,:]

        v_body0, tensor_dict0 = body(
            shape=tf.reshape(shape[:,0,:], [-1,1, 10]),
            pose=tf.reshape(pose0, [-1, 1, 72]),
            translation=tf.reshape(translation0, [-1, 3]),
        )
        v_garment_offset0 = x0+tf.reshape(self.v_TGRAMENT,[-1,3])
        v_garment_skinning0 = smpl.LBS()(v_garment_offset0, tensor_dict0["joint_transforms"], self.g_skinweight)
        v_garment_skinning0 += translation0[:, tf.newaxis, :]
        v_garment_skinning0 = tf.reshape(v_garment_skinning0,[-1,verts_num,3])

        v_body1, tensor_dict1 = body(
            shape=tf.reshape(shape[:,1,:], [-1, 1, 10]),
            pose=tf.reshape(pose1, [-1, 1, 72]),
            translation=tf.reshape(translation1, [-1, 3]),
        )
        v_garment_offset1 = x1 + tf.reshape(self.v_TGRAMENT, [-1, 3])
        v_garment_skinning1 = smpl.LBS()(v_garment_offset1, tensor_dict1["joint_transforms"], self.g_skinweight)
        v_garment_skinning1 += translation1[:, tf.newaxis, :]
        v_garment_skinning1 = tf.reshape(v_garment_skinning1, [-1, verts_num, 3])

        v_body2, tensor_dict2 = body(
            shape=tf.reshape(shape[:,2,:], [-1, 1, 10]),
            pose=tf.reshape(pose2, [-1, 1, 72]),
            translation=tf.reshape(translation2, [-1, 3]),
        )
        v_garment_offset2 = x2 + tf.reshape(self.v_TGRAMENT, [-1, 3])
        v_garment_skinning2 = smpl.LBS()(v_garment_offset2, tensor_dict2["joint_transforms"], self.g_skinweight)
        v_garment_skinning2 += translation2[:, tf.newaxis, :]
        v_garment_skinning2 = tf.reshape(v_garment_skinning2, [-1, verts_num, 3])



        return v_garment_skinning0, v_garment_skinning1, v_garment_skinning2,v_body2,tensor_dict2

def compute_L_inertia(t, f, x1, x2, x3):
    # t:帧间隔
    # x1,x2,x3: n_vertex*3
    # f: n_face*3
    #计算M
    density = 426#kg/m^3
    thick = 0.47e-3
    M = np.zeros(x1.shape[0]*3)
    for i in range(0, f.shape[0]):###########
        a = x3[f[i][1]]-x3[f[i][0]]
        b = x3[f[i][2]]-x3[f[i][1]]
        c = x3[f[i][0]]-x3[f[i][2]]
        l1 = np.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])
        l2 = np.sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2])
        l3 = np.sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2])
        s = (l1+l2+l3)/2
        S = (s*(s-l1)*(s-l2)*(s-l3))**0.5
        f_mass = density*thick*S
        for k in range(0,3):
            # for j in range(0, 3):
            #     M[3*f[i][k]+j] += f_mass/3
            M[3*f[i][k]:3*f[i][k]+3] += f_mass/3
    velocity = (x2-x1)/t
    x_hat = x2+velocity*t
    # M = tf.convert_to_tensor(np.diag(M))
    x3 = tf.reshape(x3,shape=[-1,1])
    x_hat = tf.reshape(x_hat,shape=[-1,1])
    # L = np.matmul(np.matrix(x3-x_hat),np.matrix(M),np.matrix(x3-x_hat))/(2*t**2)
    # xTM = tf.matmul(x3-x_hat,M,transpose_a=True)
    # L = tf.matmul(xTM,(x3-x_hat))
    L=0
    for i in range(0,M.shape[0]):
        x = (x3[i]-x_hat[i])**2
        x = x*M[i]
        L+=(x3[i]-x_hat[i])**2*M[i]
    L = L/(2*t**2)
    return L


def compute_L_inertia_simplify(t,x1,x2,x3,surface):
    density = 426  # kg/m^3
    thick = 0.47e-3
    vertex_mass = surface*density*thick/x3.shape[-2]
    verts_velocity = (x2-x1)/t
    x_hat = x2+verts_velocity*t
    x_bias_sum = tf.reduce_sum((x3-x_hat)**2,-1)
    L = tf.reduce_sum(x_bias_sum*vertex_mass/(2*t**2),-1)
    return L


def compute_L_strain(X, x, f):
    # x: B x V x 3
    lamda = 20.9
    mu = 11.1
    L = 0
    thick = 0.47e-3
    u = tf.gather(x, f[:, 2], axis=-2) - tf.gather(x, f[:, 0], axis=-2)
    v = tf.gather(x, f[:, 1], axis=-2) - tf.gather(x, f[:, 0], axis=-2)
    areas = tf.norm(tf.linalg.cross(u, v), axis=-1)/2.0
    x1 = tf.gather(x, indices=f[:,0], axis=-2)
    x2 = tf.gather(x, indices=f[:, 1], axis=-2)
    x3 = tf.gather(x, indices=f[:, 2], axis=-2)
    x_M = tf.concat([tf.expand_dims(x1,-2),tf.expand_dims(x2,-2),tf.expand_dims(x3,-2)],axis=-2)
    x_M = tf.transpose(x_M,[0,1,3,2])
    X1 = tf.gather(X, indices=f[:, 0], axis=-2)
    X2 = tf.gather(X, indices=f[:, 1], axis=-2)
    X3 = tf.gather(X, indices=f[:, 2], axis=-2)
    X_M = tf.concat([tf.expand_dims(X1,-2), tf.expand_dims(X2,-2), tf.expand_dims(X3,-2)], axis=-2)
    X_M = tf.transpose(X_M, [0, 2, 1])
    F = tf.matmul(x_M, tf.linalg.inv(X_M))
    G = 0.5*(tf.matmul(F,F,transpose_a=True)-tf.eye(3,3,batch_shape=[tf.constant(F.shape[0]),tf.constant(F.shape[1])]))
    phi_S = lamda/2*tf.linalg.trace(G)**2+mu*tf.linalg.trace(G@G)
    L = tf.reduce_mean(thick*areas*phi_S,axis=-1)
    return L
    for i in range(0, f.shape[0]):
        x1 = tf.reshape(x[f[i][0]],shape=[3,1])
        x2 = tf.reshape(x[f[i][1]],shape=[3,1])
        x3 = tf.reshape(x[f[i][2]],shape=[3,1])
        X1 = tf.reshape(X[f[i][0]],shape=[3,1])
        X2 = tf.reshape(X[f[i][1]],shape=[3,1])
        X3 = tf.reshape(X[f[i][2]],shape=[3,1])
        # X_M = np.matrix(np.concatenate((X1,X2,X3),axis=1))
        # x_M = np.matrix(np.concatenate((x1,x2,x3),axis=1))
        X_M = tf.concat([X1,X2,X3],axis=1)
        x_M = tf.concat([x1,x2,x3],axis=1)
        # F = np.matmul(x_M, np.linalg.inv(X_M))
        # G = 0.5*(np.matmul(np.transpose(F),F)-np.eye(3))
        # phi_S = lamda/2*np.trace(G)**2+mu*np.trace(np.matmul(G,G))
        F = tf.matmul(x_M, tf.convert_to_tensor(np.linalg.inv(X_M.numpy())))
        G = 0.5*(tf.matmul(F,F,transpose_a=True)-tf.eye(3))
        phi_S = lamda/2*np.trace(G.numpy())**2+mu*np.trace(tf.matmul(G,G).numpy())
        a = x[f[i][1]] - x[f[i][0]]
        b = x[f[i][2]] - x[f[i][1]]
        c = x[f[i][0]] - x[f[i][2]]
        l1 = np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
        l2 = np.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])
        l3 = np.sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2])
        s = (l1 + l2 + l3) / 2
        S = (s * (s - l1) * (s - l2) * (s - l3)) ** 0.5
        L += S*thick*phi_S
    return L

def compute_L_bending(x, F, neighF):
    k_bending = 3.96e-5
    # compute face normals
    V02 = tf.gather(x, F[:, 2], axis=-2) - tf.gather(x, F[:, 0], axis=-2)  # B x N x 3
    V01 = tf.gather(x, F[:, 1], axis=-2) - tf.gather(x, F[:, 0], axis=-2)  # B x N x 3
    N = tf.linalg.cross(V02, V01)  # B x N x 3
    N = N / tf.linalg.norm(N, axis=-1, keepdims=True)  # B x N x 3
    # compare neighbouring face normals
    N0 = tf.gather(N, neighF[:, 0], axis=-2)  # B x N x 3
    N1 = tf.gather(N, neighF[:, 1], axis=-2)  # B x N x 3
    # cosinus distances
    cos = tf.reduce_sum(tf.multiply(N0, N1), -1)
    err = tf.reduce_mean(1 - cos,-1)
    return err


def compute_L_gravity(v, f):
    density = 426  # kg/m^3
    thick = 0.47e-3
    M = tf.zeros([v.shape[0]])
    face_v0 = tf.gather(v, indices=f[:,0],axis=0)
    face_v1 = tf.gather(v, indices=f[:,1],axis=0)
    face_v2 = tf.gather(v, indices=f[:,2],axis=0)
    a = face_v1-face_v0
    b = face_v2-face_v1
    c = face_v0-face_v2
    l1 = tf.sqrt(tf.reduce_sum(tf.multiply(a,a),axis=1))
    l2 = tf.sqrt(tf.reduce_sum(tf.multiply(b, b),axis=1))
    l3 = tf.sqrt(tf.reduce_sum(tf.multiply(c, c),axis=1))
    s = (l1+l2+l3)/2
    S = tf.sqrt(s*(s-l1)*(s-l2)*(s-l3))
    face_mass = density*thick*S
    for i in range(0, f.shape[0]):
        for k in range(0,3):
            M[f[i][k]]+=face_mass[i]/3
    # for i in range(0, f.shape[0]):
    #     a = v[f[i][1]] - v[f[i][0]]
    #     b = v[f[i][2]] - v[f[i][1]]
    #     c = v[f[i][0]] - v[f[i][2]]
    #     l1 = np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    #     l2 = np.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])
    #     l3 = np.sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2])
    #     s = (l1 + l2 + l3) / 2
    #     S = (s * (s - l1) * (s - l2) * (s - l3)) ** 0.5
    #     f_mass = density * thick * S
    #     for k in range(0,3):
    #         M[f[i][k]] += f_mass/3
    L = 0
    g = 9.8
    L+=-1*g*tf.reduce_sum(tf.multiply(M,v[:,2]))
    return L

def compute_L_gravity_simplify(x, surface):
    density = 426
    thick = 0.47e-3
    g = 9.81
    vertex_mass = surface*density*thick/x.shape[-2]
    U = vertex_mass * g * x[:, :,2]
    return tf.reduce_sum(U,-1)

def compute_L_collision(v_cloth, v_body_normal, v_body):
    epsilon=2e-3
    k_collision=250
    L = 0
    # closest = find_nearest_neighbour(v_cloth, v_body[0])
    dist = euclidean_dist(v_cloth, v_body,batch=False)
    dist_value = tf.reduce_min(dist,axis=-1)
    dist_indices = tf.argmin(dist, axis=-1)
    del dist
    gc.collect()
    dist_mask = (dist_value < epsilon)
    collision_cloth_v = tf.boolean_mask(v_cloth, mask=dist_mask)
    collision_body_v = tf.gather(v_body,indices=tf.boolean_mask(dist_indices,mask=dist_mask),axis=-2)
    collision_body_v_normal = tf.gather(v_body_normal,indices=tf.boolean_mask(dist_indices,mask=dist_mask),axis=-2)
    collision_dot = tf.reduce_sum((tf.multiply(collision_body_v-collision_cloth_v, collision_body_v_normal)),axis=-1)
    L += k_collision*tf.reduce_mean(tf.boolean_mask(collision_dot, mask=(collision_dot>0)), axis=-1)
    return L

class Loss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Loss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # inputs: T_cloth, v1_cloth, v2_cloth, v3_cloth, f_cloth, v_body, f_body
        T_cloth, v1_cloth, v2_cloth, v3_cloth, f_cloth, v_body, f_body = inputs
        vertex_normal = smpl.VertexNormals()
        face_normal = smpl.FaceNormals()
        v3_cloth_normal = vertex_normal(tf.reshape(v3_cloth,[1,-1,3]), tf.reshape(f_cloth,[-1,3]))
        f_cloth_normal = face_normal(tf.reshape(v3_cloth,[1,-1,3]), tf.reshape(f_cloth,[1,-1,3]))
        L_inertia = compute_L_inertia_simplify(1/30,v1_cloth,v2_cloth,v3_cloth)
        L_strain = compute_L_strain(T_cloth, v3_cloth, f_cloth)
        L_bending = compute_L_bending(v3_cloth,f_cloth,outfit._neigh_F)
        L_gravity = compute_L_gravity_simplify(v3_cloth,outfit._total_area)
        L_collision = compute_L_collision(v3_cloth, v3_cloth_normal, v_body)
        L = L_inertia+L_strain+L_bending+L_gravity+L_collision
        return L

model = Deformation_model(v_Tgarment.shape[0]*3)
optimizer = tf.optimizers.SGD(learning_rate=0.001)
# def train_process():
#     poses, trans, trans_vel = utils.load_motion(args.motion)
#     epochs = 10
#     batch_size = 16######16
#     batch_num = math.ceil(poses.shape[0]/batch_size)
#     hidden_states = [
#         tf.random.normal([1,256],mean=0,stddev=0.1, dtype=tf.float32),  # State 0
#         tf.random.normal([1,256],mean=0,stddev=0.1, dtype=tf.float32),  # State 1
#         tf.random.normal([1,256],mean=0,stddev=0.1, dtype=tf.float32),  # State 2
#         tf.random.normal([1,256],mean=0,stddev=0.1, dtype=tf.float32),  # State 3
#     ]
#     for epoch in range(0, epochs):
#         for batch_iter in range(0, batch_num):
#             loss = 0
#             with tf.GradientTape() as tape:
#                 for batch in range(0,batch_size):
#                     frame = batch_iter*batch_size+batch
#                     if frame>=poses.shape[0]:
#                         break
#                     pose = tf.reshape(poses[frame], [1, 1, 72])
#                     shape = tf.reshape(betas, (1, 1, 10))
#                     translation = tf.reshape(trans[frame], (1, 1, 3))
#                     translation_vel = tf.reshape(trans_vel[frame], (1, 1, 3))
#                     v_body, tensor_dict = body(
#                         shape=tf.reshape(shape, [-1, 10]),
#                         pose=tf.reshape(pose, [-1, 72]),
#                         translation=tf.reshape(translation, [-1, 3]),
#                     )
#                     x1 = model([pose,translation_vel,shape,hidden_states[0],hidden_states[1],hidden_states[2],hidden_states[3],\
#                                 translation,tensor_dict["shape_blendshape"],tensor_dict["joint_transforms"],tensor_dict["vertices"],tensor_dict["vertex_normals"]])
#                     frame = batch_iter * batch_size + batch+1
#                     if frame>=poses.shape[0]:
#                         break
#                     pose = tf.reshape(poses[frame], [1, 1, 72])
#                     shape = tf.reshape(betas, (1, 1, 10))
#                     translation = tf.reshape(trans[frame], (1, 1, 3))
#                     translation_vel = tf.reshape(trans_vel[frame], (1, 1, 3))
#                     v_body, tensor_dict = body(
#                         shape=tf.reshape(shape, [-1, 10]),
#                         pose=tf.reshape(pose, [-1, 72]),
#                         translation=tf.reshape(translation, [-1, 3]),
#                     )
#                     x2 = model([pose, translation_vel, shape, hidden_states[0], hidden_states[1], hidden_states[2],
#                                hidden_states[3], \
#                                translation, tensor_dict["shape_blendshape"], tensor_dict["joint_transforms"],
#                                tensor_dict["vertices"], tensor_dict["vertex_normals"]])
#                     frame = batch_iter * batch_size + batch+2
#                     if frame>=poses.shape[0]:
#                         break
#                     pose = tf.reshape(poses[frame], [1, 1, 72])
#                     shape = tf.reshape(betas, (1, 1, 10))
#                     translation = tf.reshape(trans[frame], (1, 1, 3))
#                     translation_vel = tf.reshape(trans_vel[frame], (1, 1, 3))
#                     v_body, tensor_dict = body(
#                         shape=tf.reshape(shape, [-1, 10]),
#                         pose=tf.reshape(pose, [-1, 72]),
#                         translation=tf.reshape(translation, [-1, 3]),
#                     )
#                     x3 = model([pose, translation_vel, shape, hidden_states[0], hidden_states[1], hidden_states[2],
#                                hidden_states[3], \
#                                translation, tensor_dict["shape_blendshape"], tensor_dict["joint_transforms"],
#                                tensor_dict["vertices"], tensor_dict["vertex_normals"]])
#                     loss += Loss()([v_Tgarment, x1, x2, x3, f_garment, v_body, body.faces])
#                 loss = loss/batch_size
#                 loss = tf.convert_to_tensor(loss)
#             grads = tape.gradient(loss, model.trainable_variables)
#             optimizer.apply_gradients(zip(grads, model.trainable_variables))


def train():
    batch_size = 1
    tr_data = Data.Data(batch_size=batch_size)
    tr_steps = np.floor(tr_data._n_samples / batch_size)
    num_epochs = 10
    hidden_states = [
                tf.random.normal([batch_size,256],mean=0,stddev=0.1, dtype=tf.float32),  # State 0
                tf.random.normal([batch_size,256],mean=0,stddev=0.1, dtype=tf.float32),  # State 1
                tf.random.normal([batch_size,256],mean=0,stddev=0.1, dtype=tf.float32),  # State 2
                tf.random.normal([batch_size,256],mean=0,stddev=0.1, dtype=tf.float32),  # State 3
            ]
    save_model_dir = 'train_model_save'
    for epoch in range(num_epochs):
        print("")
        print("Epoch " + str(epoch + 1))
        print("--------------------------")
        """ TRAIN """
        print("Training...")
        betas = np.zeros(10, dtype=np.float32)
        shape = tf.reshape(betas, (1, 1, 10))
        total_time = 0
        step = 0
        metrics = [0] * 4  # Edge, Bend, Gravity, Collisions
        start = time.time()
        batch_iter = 0
        for poses, trans, trans_vel in tr_data._iterator:
            """ Train step """
            with tf.GradientTape() as tape:
                pred1,pred2,pred3,v_body,body_tensor_dict = model([shape, poses, trans, trans_vel, hidden_states[0], hidden_states[1],\
                             hidden_states[2], hidden_states[3]])
                L_inertia = compute_L_inertia_simplify(1/30,pred1,pred2,pred3,outfit._total_area)
                L_strain = compute_L_strain(v_Tgarment,pred3,f_garment)
                L_bend = compute_L_bending(pred3,f_garment,outfit._neigh_F)
                L_gravity = compute_L_gravity_simplify(pred3,outfit._total_area)
                L_collision = 0
                for i in range(0, pred3.shape[0]):
                    # 整个batch算距离矩阵时分配不出足够空间
                    L_collision += compute_L_collision(pred3[i],body_tensor_dict["vertex_normals"][i],v_body[i])
                loss = tf.reduce_mean(L_inertia+L_strain+L_gravity+L_bend,axis=-1)+L_collision/batch_size
            """ Backprop """
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            print("epoch{}, batch{}: ".format(epoch,batch_iter))
            print("L_inertia={},L_strain={},L_bend={},L_gravity={},L_collision={}".format(\
                tf.reduce_mean(L_inertia),tf.reduce_mean(L_strain),tf.reduce_mean(L_bend),\
                tf.reduce_mean(L_gravity),tf.reduce_mean(L_collision)))
            batch_iter+=1
        model_save_path = os.path.join(save_model_dir,'{}.h5'.format(epoch))
        model.save(model_save_path)

train()
# if __name__ == "__main__":


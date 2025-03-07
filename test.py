import numpy as np
from scipy.sparse import diags
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix

# cube = np.arange(1, 28, 1)
# print(cube)
# print(cube.reshape((3,3,3)))




def stamp_mats3d(rows, cols, depths, mat1, mat2):
    mat = np.zeros((rows*cols*depths, rows*cols*depths))

    diag_mat = stamp_mats(rows, cols, get_mat1(rows, cols), get_mat2(rows, cols))

    off_diag_mat = stamp_mats(rows, cols, get_mat2(rows, cols), get_mat2(rows, cols))
    small_mat = diags([np.ones(cols-1),np.ones(cols),np.ones(cols-1)], [-1,0,1]).toarray()
    # print(off_diag_mat)
    # small_mat[0][1] = 0
    # off_diag_mat[0:col,-col-1:-1] = small_mat

    for i in range(0, rows*cols*depths-rows*cols, rows*cols):
        mat[i+rows*cols:(i+2*rows*cols),i:(i+rows*cols)]=off_diag_mat
        mat[i:(i+rows*cols),i+rows*cols:(i+2*rows*cols)]=off_diag_mat

    for i in range(0, rows*cols*depths, rows*cols):
        mat[i:(i+rows*cols),i:(i+rows*cols)]=diag_mat

    return mat


def get_mat1(rows, cols):
    return diags([np.ones(cols-1), np.ones(cols-1)],[-1,1]).toarray()

def get_mat2(rows, cols):
    return diags([np.ones(cols-1),np.ones(cols),np.ones(cols-1)], [-1,0,1]).toarray()

# def create_off_diag_mat(size, upper_size):
#     size=int(size)
#     mat = np.zeros((upper_size, upper_size))
#     reference = get_mat2(size, size)
#     for i in range(0, upper_size-size, size):
#         mat[i+size:(i+2*size),i:(i+size)]=reference
#         mat[i:(i+size),i+size:(i+2*size)]=reference

#     # pasting the diagonal interior matrices
#     for i in range(0,upper_size, size):
#         mat[i:(i+size),i:(i+size)]=reference

#     return mat

def get_2D_adj(rows, cols, mat1, mat2):
    mat = np.zeros((rows*cols, rows*cols))
    for i in range(0, rows*cols-cols, cols):
        mat[i+cols:(i+2*cols),i:(i+cols)]=mat2
        mat[i:(i+cols),i+cols:(i+2*cols)]=mat2

    # pasting the diagonal interior matrices
    for i in range(0,rows*cols, cols):
        mat[i:(i+cols),i:(i+cols)]=mat1

    return mat

def create_1D_adj_mat(size):
    return diags([np.ones(size-1), np.ones(size-1)],[-1,1]).toarray()

# def create_n_dim_adj_mat(dim, size):
#     side_length = np.power(size, dim)
#     lower_length = int(side_length/size)

#     if dim == 1:
#         # if user specifically asks for 1D
#         return create_1D_adj_mat(side_length)
#     elif dim == 2:
#         # for anything above 1D
#         return get_2D_adj(size, size, get_mat1(size, size), get_mat2(size, size))
    
#     mat = np.zeros((side_length, side_length))

#     off_diag_mat = create_off_diag_mat(int(lower_length/size), lower_length)
#     plt.matshow(off_diag_mat)
#     plt.show()

#     # along off-diagonal
#     for i in range(0, int(side_length-lower_length), int(lower_length)):
#         mat[i+lower_length:(i+2*lower_length),i:(i+lower_length)]=off_diag_mat
#         mat[i:(i+lower_length),i+lower_length:(i+2*lower_length)]=off_diag_mat
    

#     # along the diagonal
#     for i in range(0,side_length, lower_length):
#         mat[i:(i+lower_length),i:(i+lower_length)]=create_n_dim_adj_mat(dim-1, size)

#     return mat

def create_off_diag(side_length, size):

    if side_length == size:
        return get_mat2(size, size)

    mat = np.zeros((side_length, side_length))
    lower_size = int(side_length/size)

    # paste smaller matrix along diag and off-diag

    reference = create_off_diag(lower_size, size)

    for i in range(0, side_length-lower_size, lower_size):
        mat[i+lower_size:(i+2*lower_size),i:(i+lower_size)]=reference
        mat[i:(i+lower_size),i+lower_size:(i+2*lower_size)]=reference

    # pasting the diagonal interior matrices
    for i in range(0,side_length, lower_size):
        mat[i:(i+lower_size),i:(i+lower_size)]=reference

    return mat

def create_n_dim_adj_mat(dim, size):
    mat_side_length = np.power(size, dim)

    if dim == 1:
        # if user specifically asks for 1D
        return create_1D_adj_mat(side_length)
    elif dim == 2:
        # for anything above 1D
        return get_2D_adj(size, size, get_mat1(size, size), get_mat2(size, size))
    
    # create the matrix
    mat = np.zeros((mat_side_length, mat_side_length))

    lower_side_length = int(mat_side_length/size)
    off_diag_mat = create_off_diag(lower_side_length, size)

    # paste along the off-diagonal on both sides (symmetric matrix so no need to transpose)
    for i in range(0, mat_side_length-lower_side_length, lower_side_length):
        mat[i+lower_side_length:(i+2*lower_side_length),i:(i+lower_side_length)]=off_diag_mat
        mat[i:(i+lower_side_length),i+lower_side_length:(i+2*lower_side_length)]=off_diag_mat
    
    # paste along the diagonal
    for i in range(0, mat_side_length, lower_side_length):
        mat[i:(i+lower_side_length),i:(i+lower_side_length)]=create_n_dim_adj_mat(dim-1, size)
    
    return csr_matrix(mat)




# # 1D 1x5
# line = np.arange(1,6,1)
# adj_mat = get_mat1(1,5)

# # print(line)
# # plt.matshow(adj_mat)
# # plt.show()


# # 2D 5x5
# grid = np.arange(1,26,1)
# mat1 = get_mat1(5,5)
# mat2 = get_mat2(5,5)

# adj_mat = stamp_mats(5, 5, mat1, mat2)

# print(grid.reshape((5,5)))
# plt.matshow(adj_mat)
# plt.show()


# g = np.array([[0, 1, 0, 1, 1, 0, 0, 0, 0], [1, 0, 1, 1, 1, 1, 0, 0, 0], [0, 1, 0, 0, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 0, 1, 1, 0], [1, 1, 1, 1, 0, 1, 1, 1, 1], [0, 1, 1, 0, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 0, 0, 1, 0], [0, 0, 0, 1, 1, 1, 1, 0, 1], [0, 0, 0, 0, 1, 1, 0, 1, 0]])
# rows = 5
# cols = 4
# g = stamp_mats(rows, cols, get_mat1(rows, cols),get_mat2(rows, cols))

# # plt.matshow(g)
# # plt.show()

# colors = np.zeros(rows*cols)

# site = 19
# row = g[site]
# for i in range(len(row)):
#     if row[i] == 1:
#         colors[i] = 1

# plt.imshow(np.reshape(colors, (rows,cols), 'C'), cmap='binary')
# plt.show()


# # 3D 5x5x5

# row, col, depth = 5,5,5

# cube = np.arange(1,row*col*depth+1,1)
# tdcube = cube.reshape((row,col,depth))

# mat1 = get_mat1(row,col)
# mat2 = get_mat2(row,col)
# adj_mat = stamp_mats3d(row, col, depth, mat1, mat2)
# # plt.matshow(adj_mat)
# # plt.show()
# # print(adj_mat[4])



# adj_mat = create_n_dim_adj_mat(3, 10)


# plt.matshow(adj_mat)
# plt.show()

# colors = ['blue' for i in range(row*col*depth)]
# site = 30
# for i in range(len(adj_mat[site])):
#     if adj_mat[site][i] == 1:
#         colors[i] = "yellow"


# colors = np.array(colors)
# col = colors.reshape(row,col,depth)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plotting the random binary voxels
# ax.voxels(tdcube, facecolors=col, edgecolor='k', alpha=0.6)

# # Setting labels
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')

# plt.title('Basic 3D Voxels')
# plt.show()


import keras
import numpy as np
import os
import copy

def permutate_labels(labels, path='2_label_permutation.npy'):
	order = np.load(path)
	labs = [order[i] for i in labels]
	return np.array(labs)

def load_data(path, nb_labels=-1, one_hot=False):
	labels = np.load('data/mnist_labels.npy').astype(np.int)
	if one_hot:
		labels = keras.utils.to_categorical(labels, nb_labels)
	#labs = np.load('mnist_labels.npy')
	#label_permutation = np.load('2_label_permutation.npy')[:int(nb_labels)].T
	#labels = np.array([label_permutation[i] for i in labs])
	order = np.load(path)#'256_65536_permutation.npy'
	if os.path.exists('data/{}_mnist_data.npy'.format(path.split('_')[1])):
		imgs = np.load('data/{}_mnist_data.npy'.format(path.split('_')[1]))
		input_shape = (28, 28, int(path.split('_')[1].split('.')[-1]))
	# if eval(path.split('/')[-1].split('_')[1]) == 65536:
	# 	input_shape = (28, 28, 1)
	# 	imgs = np.load('65536_mnist_data.npy')
	# elif eval(path.split('/')[-1].split('_')[1]) == 256.2:
	# 	imgs = np.load('256_2_mnist_data.npy')
	# 	input_shape = (28, 28, 2)
	elif len(order.shape) > 1:
		input_shape = (28, 28, int(path.split('_')[1].split('.')[-1]))
		imgs = np.transpose(np.load('data/mnist_data.npy').astype(np.int), (0,2,3,1))
		#imgs = np.clip(np.transpose(np.load('mnist_data.npy').astype(np.float32)+1, (1,0,2,3))[0], 0, 255).astype(np.int)

		# tmp = np.array([copy.deepcopy(imgs) for i in np.arange(int(path.split('_')[1].split('.')[-1]))])
		samples = np.array([[[order[d[0]] for d in c] for c in b] for b in imgs])
		imgs = samples.astype(np.float32) / (int(path.split('_')[1].split('.')[0])-1)
		# np.save('data/{}_mnist_data.npy'.format(path.split('_')[1]),imgs)
	elif len(order.shape) == 1:
		input_shape = (28, 28, 1)
		imgs = np.transpose(np.load('data/mnist_data.npy'), (0,2,3,1)) 
		samples = np.array([[[[order[a] for a in b] for b in c] for c in d] for d in imgs])
		imgs = samples.astype(np.float32) / (int(path.split('_')[1].split('.')[0])-1)
		# np.save('data/{}_mnist_data.npy'.format(path.split('_')[1]), imgs)
	return imgs, labels, input_shape

def extend_data(path, imgs):
	if np.max(imgs) <= 1:
		imgs *= 255
	order = np.load(path)
	imgs = imgs.astype(np.int)
	samples = np.array([[[order[d[0]] for d in c] for c in b] for b in imgs]).astype(np.float32) / (int(path.split('_')[1].split('.')[0])-1)
	return samples

def order_extend_data(order, imgs, basis=255):
	if np.max(imgs) <= 1:
		imgs *= 255
	imgs = imgs.astype(np.int)
	samples = np.array([[[order[d[0]] for d in c] for c in b] for b in imgs]).astype(np.float32) / basis
	return samples

def two_pixel_perm_img(nb_channal, imgs):
	np.random.seed(0)
	perms = []
	for j in range(256):
		perm = []
		for i in range(nb_channal):
			perm.append(np.random.permutation(np.arange(256)))
		perms.append(perm)
	perms = np.array(perms).transpose((0,2,1))

	if np.max(imgs) <= 1:
		imgs *= 255
	imgs = imgs.reshape(-1, 784).astype(np.int)
	print(imgs.shape)
	imgs = np.array([[perms[a[i]][a[i+1]] for i in range(0, len(a), 2)] for a in imgs]).reshape(-1, 28, 14, nb_channal)
	return imgs

def two_pixel_perm(nb_channal, model_dir):
	np.random.seed(0)
	perms = []
	for j in range(256):
		perm = []
		for i in range(nb_channal):
			perm.append(np.random.permutation(np.arange(256)))
		perms.append(perm)
	perms = np.array(perms).transpose((0,2,1))
	print(perms.shape)
	imgs = np.load('data/mnist_data.npy').transpose((0,2,3,1)).reshape(-1, 784)
	imgs = np.array([[perms[a[i]][a[i+1]] for i in range(0, len(a), 2)] for a in imgs]).reshape(-1, 28, 14, nb_channal)
	labels = np.load('data/mnist_labels.npy')
	input_shape = imgs.shape
	return imgs, labels, input_shape, model_dir+'_two'

def two_pixel_perm_sliding_img(nb_channal, img):
	np.random.seed(0)
	perms = []
	for j in range(256):
		perm = []
		for i in range(nb_channal):
			perm.append(np.random.permutation(np.arange(256)))
		perms.append(perm)

	perms = np.array(perms).transpose((0,2,1))
	print(perms.shape)

	if np.max(imgs) <= 1:
		imgs *= 255
	imgs = imgs.transpose((1,0,2,3))[0].astype(np.int)
	print(imgs.shape)
	imgs = np.array([[[perms[b[i-1]][b[i]] for i in range(1, len(b), 1)] for b in a] for a in imgs]).reshape(-1, 28, 14, nb_channal)
	return imgs

def two_pixel_perm_sliding(nb_channal, model_dir):
	np.random.seed(0)
	perms = []
	for j in range(256):
		perm = []
		for i in range(nb_channal):
			perm.append(np.random.permutation(np.arange(256)))
		perms.append(perm)

	perms = np.array(perms).transpose((0,2,1))
	print(perms.shape)
	imgs = np.load('data/mnist_data.npy').transpose((1,0,2,3))[0]
	imgs = np.array([[[perms[b[i-1]][b[i]] for i in range(1, len(b), 1)] for b in a] for a in imgs]).reshape(-1, 28, 14, nb_channal)
	labels = np.load('data/mnist_labels.npy')
	input_shape = imgs.shape
	return imgs, labels, input_shape, model_dir+'_slide'


def diff_perm_per_classifier(st_lab, nb_channal, model_dir):
	np.random.seed(st_lab)
	perm = []
	for i in range(nb_channal):
		perm.append(np.random.permutation(np.arange(256)))
	perm = np.array(perm).transpose((1,0))
	imgs = np.load('data/mnist_data.npy').transpose((0,2,3,1))
	imgs = order_extend_data(perm, imgs)
	labels = np.load('data/mnist_labels.npy')
	input_shape = imgs.shape[1:]
	return imgs, labels, input_shape, model_dir+'_lab'

def show_image(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))
	

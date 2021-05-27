#import imageio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def sigmoid(x):
    """
    Sigmoid activation function, range [0,1]
    Input:
        x - input training batch (numpy array)
    """
    return 1. / (1. + np.exp(-x))


def dsigmoid(x):
    """
    Derivative of sigmoid activation function
    Input:
        x - input training batch (numpy array)
    """
    y = sigmoid(x)
    return y * (1. - y)


def dtanh(x):
    """
    Tanh activation function, range [-1,1]
    Input:
        x - numpy array
    """
    return 1. - np.tanh(x) ** 2


def lrelu(x, alpha=1e-2):
    """
    Leaky ReLU activation function
    Input:
        x - numpy array
        alpha - gradient of mapping function for negative values
        if alpha = 0 then this corresponds to ReLU activation
    """
    return np.maximum(x, x * alpha)


def dlrelu(x, alpha=1e-2):
    """
    Derivative of leaky ReLU activation function
    Input:
        x - numpy array
        alpha - gradient of mapping function for negative values
        if alpha = 0 then this corresponds to the dervative of ReLU
    """
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

class GAN:
    def __init__(self, number, epochs=100, batch_size=64, input_layer_size_g=100,
                 hidden_layer_size=128, learning_rate=1e-3,
                 decay_rate=1e-4, image_size=28, display_epochs=5, create_gif=True):
        """
        Implementation of a vanilla GAN (the simplest GAN) using Numpy.
        The Generator and Discriminator are described by multilayer perceptrons.
        Input:
            number - # chosen the number to be generated
            epochs - # #training iterations
            batch_size - # #training examples in each batch
            input_layer_size_g - # #neurons in the input layer of the generator
            hidden_layer_size_g - # #neurons in the hidden layer of the generator
            hidden_layer_size_d - # #neurons in the hidden layer of the discriminator
            learning_rate - to what extent newly acquired info. overrides old info.
            decay_rate - learning rate decay after every epoch
            image_size - # of pixels of training images
            display_epochs - after how many epochs to display intermediary results
            create_gif - if true, a gif of sample images will be generated
        """
        # -------- Initialise hyperparameters --------#
        self.number = number
        self.epochs = epochs
        self.batch_size = batch_size
        self.H1 = input_layer_size_g
        self.H2 = hidden_layer_size
        self.H2 = hidden_layer_size
        self.lr = learning_rate
        self.dr = decay_rate
        self.image_size = image_size # 28
        self.display_epochs = display_epochs
        self.create_gif = create_gif

        self.image_dir = Path('./GAN_sample_images')  # new a folder in current directory
        if not self.image_dir.is_dir():
            self.image_dir.mkdir()

        self.filenames = []  # stores filenames of sample images if create_gif is enabled

        # -------- Initialise weights with Xavier method --------#
        # -------- Generator --------#
        self.G_W0 = np.random.randn(self.H1, self.H2) * np.sqrt(2. / self.H1)  # 100x128
        self.G_b0 = np.zeros(self.H2)  # 1x100
        self.G_W1 = np.random.randn(self.H2, self.image_size ** 2) * np.sqrt(2. / self.H2)  # 128x784
        self.G_b1 = np.zeros(self.image_size ** 2)  # 1x784

        # -------- Discriminator --------#
        self.D_W0 = np.random.randn(self.image_size ** 2, self.H2) * np.sqrt(2. / self.image_size ** 2)  # 784x128
        self.D_b0 = np.zeros(self.H2)  # (128, )
        self.D_W1 = np.random.randn(self.H2, 1) * np.sqrt(2. / self.H2)  # (128, 1)
        self.D_b1 = 0.0  


    def forward_generator(self, z):
        """
        Implements forward propagation through the Generator
        Input:
            z - batch of random noise from normal distribution
        Output:
            G_z2 - logit output from generator
            G_f2 - generated images
        """
        z = z.reshape(self.batch_size, -1) # in case z is 3d
        # G_z1 = np.dot(z, G_W0) + G_b0
        G_z1 = np.dot(z, self.G_W0) + self.G_b0
        # G_f1 = lrelu(G_z1)
        G_f1 = lrelu(G_z1)

        G_z2 = np.dot(G_f1, self.G_W1) + self.G_b1
        G_f2 = np.tanh(G_z2)  # check: range -1 to 1 as real images
        return G_z1, G_f1, G_z2, G_f2


    def forward_discriminator(self, x):
        """
        Implements forward propagation through the Discriminator
        Input:
            x - batch of real/fake images
        Output:
            D_z2 - logit output from discriminator D(x) / D(G(z))
            D_f2 - discriminator's output prediction for real/fake image
        """
        x = x.reshape(self.batch_size, -1)
        # D_z1 = np.dot(x, D_W0) + D_b0
        D_z1 = np.dot(x, self.D_W0) + self.D_b0
        # D_f1 = lrelu(D_z1)
        D_f1 = lrelu(D_z1)

        # D_z2 = np.dot(D_f1, D_W1) + D_b1
        D_z2 = np.dot(D_f1, self.D_W1) + self.D_b1
        # D_f2 = sigmoid(D_z2)
        D_f2 = sigmoid(D_z2)  # check: output probability between [0,1]
        return D_z1, D_f1, D_z2, D_f2


    def backward_discriminator(self, x_real, D_z1_real, D_f1_real, D_z2_real, D_f2_real, x_fake, D_z1_fake, D_f1_fake, D_z2_fake, D_f2_fake):
        """
        Implements backward propagation through the discriminator for fake & real images
        Input:
            x_real - batch of real images from training data
            D_z2_real - logit output from discriminator D(x)
            D_f2_real - discriminator's output prediction for real images
            x_fake - batch of generated (fake) images from the generator
            D_z2_fake - logit output from discriminator D(G(z))
            D_f2_fake - discriminator's output prediction for fake images
        """
        # -------- Backprop through Discriminator --------#
        # J_D = np.mean(-np.log(D_f2_real) - np.log(1 - D_f2_fake))

        # real input gradients -np.log(D_f2_real)
        dD_f2_real = -1. / (D_f2_real + 1e-8)  # 64x1
        dD_z2_real = dD_f2_real * dsigmoid(D_z2_real)  # 64x1
        dD_W1_real = np.dot(D_f1_real.T, dD_z2_real)
        dD_b1_real = np.sum(dD_z2_real, axis=0)

        dD_f1_real = np.dot(dD_z2_real, self.D_W1.T)
        dD_z1_real = dD_f1_real * dlrelu(D_z1_real)
        dD_W0_real = np.dot(x_real.T, dD_z1_real)
        dD_b0_real = np.sum(dD_z1_real, axis=0)

        # fake input gradients -np.log(1 - D_f2_fake)
        dD_f2_fake = 1. / (1. - D_f2_fake + 1e-8)
        dD_z2_fake = dD_f2_fake * dsigmoid(D_z2_fake)
        dD_W1_fake = np.dot(D_f1_fake.T, dD_z2_fake)
        dD_b1_fake = np.sum(dD_z2_fake, axis=0)

        dD_f1_fake = np.dot(dD_z2_fake, self.D_W1.T)
        dD_z1_fake = dD_f1_fake * dlrelu(D_z1_fake)
        dD_W0_fake = np.dot(x_fake.T, dD_z1_fake)
        dD_b0_fake = np.sum(dD_z1_fake, axis=0)

        # -------- Combine gradients for real & fake images--------#
        dD_W1 = dD_W1_real + dD_W1_fake
        dD_b1 = dD_b1_real + dD_b1_fake

        dD_W0 = dD_W0_real + dD_W0_fake
        dD_b0 = dD_b0_real + dD_b0_fake

        # -------- Update gradients using SGD--------#
        self.D_W0 -= self.lr * dD_W0
        self.D_b0 -= self.lr * dD_b0

        self.D_W1 -= self.lr * dD_W1
        self.D_b1 -= self.lr * dD_b1


    def backward_generator(self, z, G_z1, G_f1, G_z2, x_fake, D_z1_fake, D_f1_fake, D_z2_fake, D_f2_fake):
        """
        Implements backward propagation through the Generator
        Input:
            z - random noise batch
            x_fake - batch of generated (fake) images
            D_z2_fake - logit output from discriminator D(G(z))
            D_f2_fake - output prediction from discriminator D(G(z))
        """
        # -------- Backprop through Discriminator --------#
        # J_D = np.mean(-np.log(D_f2_real) - np.log(1 - D_f2_fake))

        # fake input gradients -np.log(1 - D_f2_fake)
        dD_f2_fake = -1. / (D_f2_fake + 1e-8)  # 64x1
        dD_z2_fake = dD_f2_fake * dsigmoid(D_z2_fake)
        dD_f1_fake = np.dot(dD_z2_fake, self.D_W1.T)
        dD_z1_fake = dD_f1_fake * dlrelu(D_z1_fake)
        dx_d = np.dot(dD_z1_fake, self.D_W0.T)

        # -------- Backprop through Generator --------#
        # J_G = np.mean(-np.log(1 - D_f2_fake))
        # fake input gradients -np.log(1 - D_f2_fake)
        dG_z2 = dx_d * dtanh(G_z2)
        dG_W1 = np.dot(G_f1.T, dG_z2)
        dG_b1 = np.sum(dG_z2, axis=0)

        dG_f1 = np.dot(dG_z2, self.G_W1.T)
        dG_z1 = dG_f1 * dlrelu(G_z1, alpha=0)
        dG_W0 = np.dot(z.T, dG_z1)
        dG_b0 = np.sum(dG_z1, axis=0)

        # -------- Update gradients using SGD --------#
        self.G_W0 -= self.lr * dG_W0
        self.G_b0 -= self.lr * dG_b0

        self.G_W1 -= self.lr * dG_W1
        self.G_b1 -= self.lr * dG_b1


    def preprocess_data(self, x, y):
        """
        Processes the training images and labels:
            1. Only includes samples relevant for training
            i.e. chosen by the user in the numbers list
            2. Removes samples that can't be in a full batch
            3. Scales the images to the range of tanh [-1,1]
            4. Shuffles the data to enable convergence
        Input:
            x - raw training images
            y - raw training labels
        Output:
            x_train - processed training images
            y_train - processed training labels
            num_batches - number of batches
        """
        x_train = []
        y_train = []

        # limit the data to a subset of digits from 0-9
        for i in range(y.shape[0]):
            if y[i] in self.number:
                x_train.append(x[i])
                y_train.append(y[i])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # limit the data to full batches only
        num_batches = x_train.shape[0] // self.batch_size
        x_train = x_train[: num_batches * self.batch_size]
        y_train = y_train[: num_batches * self.batch_size]

        # flatten the images (_,28,28)->(_, 784)
        x_train = np.reshape(x_train, (x_train.shape[0], -1))

        # normalise the data to the range [-1,1]
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5

        # shuffle the data
        idx = np.random.permutation(len(x_train))
        x_train, y_train = x_train[idx], y_train[idx]
        return x_train, y_train, num_batches


    def sample_images(self, images, epoch, show):
        """
        Generates a grid with sample images from the generator.
        Images are stored in the GAN_sample_images folder in the local directory.
        Input:
            images - generated images (numpy array)
            epoch - current training iteration, used to identify images
            show - if True, the grid of images is displayed
        """
        images = np.reshape(images, (self.batch_size, self.image_size, self.image_size))

        fig = plt.figure(figsize=(4, 4))

        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(images[i] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        # saves generated images in the GAN_sample_images folder
        if self.create_gif:
            current_epoch_filename = self.image_dir.joinpath(f"GAN_epoch{epoch}.png")
            self.filenames.append(current_epoch_filename)
            plt.savefig(current_epoch_filename)

        if show == True:
            plt.show()
        else:
            plt.close()


    #def generate_gif(self):
    #    """
    #    Generates a gif from the exported generated images at each training iteration
    #    Input:
    #        filename - name of gif, stored in local directory
    #    """
    #    images = []
    #    for filename in self.filenames:
    #        images.append(imageio.imread(filename))
    #    imageio.mimsave("GAN.gif", images)


    def train(self, x, y):
        """
        Main method of the GAN class where training takes place
        Input:
            x - training data, size [no. samples, 28, 28] (numpy array)
            y - training labels, size [no.samples, 1] (numpy array)
        Output:
            J_Ds - Discriminator loss for each pass through the batches (list)
            J_Gs - Generator loss for each pass through the batches(list)
        """
        J_Ds = []  # stores the disciminator losses
        J_Gs = []  # stores the generator losses

        # preprocess input; note that labels aren't needed
        x_train, _, num_batches = self.preprocess_data(x, y)

        for epoch in range(self.epochs):
            for i in range(num_batches):
                # ------- PREPARE INPUT BATCHES & NOISE -------#
                x_real = x_train[i * self.batch_size: (i + 1) * self.batch_size]
                z = np.random.normal(0, 1, size=[self.batch_size, self.H1])  # 64x100

                # ------- FORWARD PROPAGATION -------#
                G_z1, G_f1, G_z2, x_fake = self.forward_generator(z)

                D_z1_real, D_f1_real, D_z2_real, D_f2_real = self.forward_discriminator(x_real)
                D_z1_fake, D_f1_fake, D_z2_fake, D_f2_fake = self.forward_discriminator(x_fake)

                # ------- CROSS ENTROPY LOSS -------#
                # ver1 : max log(D(x)) + log(1 - D(G(z))) (in original paper)
                # ver2 : min -log(D(x)) min log(1 - D(G(z))) (implemented here)
                J_D = np.mean(-np.log(D_f2_real) - np.log(1 - D_f2_fake))
                J_Ds.append(J_D)

                # ver1 : minimize log(1 - D(G(z))) (in original paper)
                # ver2 : maximize log(D(G(z)))
                # ver3 : minimize -log(D(G(z))) (implemented here)
                J_G = np.mean(-np.log(D_f2_fake))
                J_Gs.append(J_G)
                # ------- BACKWARD PROPAGATION -------#
                self.backward_discriminator(x_real, D_z1_real, D_f1_real, D_z2_real, D_f2_real,
                                            x_fake, D_z1_fake, D_f1_real, D_z2_fake, D_f2_fake)
                self.backward_generator(z, G_z1, G_f1, G_z2, x_fake, D_z1_fake, D_f1_fake, D_z2_fake, D_f2_fake)

            if epoch % self.display_epochs == 0:
                print(
                    f"Epoch:{epoch:}|G loss:{J_G:.4f}|D loss:{J_D:.4f}|D(G(z))avg:{np.mean(D_f2_fake):.4f}|D(x)avg:{np.mean(D_f2_real):.4f}|LR:{self.lr:.6f}")
                self.sample_images(x_fake, epoch, show=True)  # display sample images
            else:
                self.sample_images(x_fake, epoch, show=False)

            # reduce learning rate after every epoch
            self.lr = self.lr * (1.0 / (1.0 + self.dr * epoch))

        # generate gif
        if self.create_gif:
            self.generate_gif()
        return J_Ds, J_Gs
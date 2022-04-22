import matplotlib.pyplot as plt

def draw_losses(output_file, title_prefix=None):
    d_target_loss = []
    d_source_loss = []
    adv_loss = []
    f = open(output_file, 'r')
    for line in f:
        if ": [d_target loss: " in line:
            spl = line.split(' ')
            d_target_loss.append(float(spl[3][:-1]))
            d_source_loss.append(float(spl[6][:-1]))
            adv_loss.append(float(spl[9][:-1]))
    plt.plot(d_target_loss)
    plt.plot(d_source_loss)
    plt.title(title_prefix + 'Discriminators loss')
    plt.show()
    plt.plot(adv_loss)
    plt.title(title_prefix + 'Adversarial loss')
    plt.show()

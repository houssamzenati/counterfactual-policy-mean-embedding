########### VIZUALIZATION HELPERS #######################
import numpy as np
import matplotlib.pyplot as plt

def perc(data):
    median = np.zeros(data.shape[1])
    perc_25 = np.zeros(data.shape[1])
    perc_75 = np.zeros(data.shape[1])
    std_data = np.zeros(data.shape[1])
    for i in range(0, len(median)):
        median[i] = np.mean(data[:, i])
        perc_25[i] = np.percentile(data[:, i], 25)
        perc_75[i] = np.percentile(data[:, i], 75)
        std_data[i] = np.std(data[:, i])

    return median, perc_25, perc_75, std_data


def subplot_1D_signals(
    X, title="", title_fontsize=20, figsize=(10, 5), linewidth=1, colorcode="#050C12"
):
    """Plot the 1D signals (each row from the given matrix)"""
    n = X.shape[0]  # Number of signals

    fig, ax = plt.subplots(n, 1, figsize=figsize)

    for i in range(n):
        ax[i].plot(X[i, :], linewidth=linewidth, color=colorcode)
        ax[i].grid()

    plt.suptitle(title, fontsize=title_fontsize)
    # plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
    plt.draw()

def substem_1D_signals(
    X, title="", title_fontsize=20, figsize=(10, 5)
):
    """Stem plot the 1D signals (each row from the given matrix)"""
    n = X.shape[0]  # Number of signals

    fig, ax = plt.subplots(n, 1, figsize=figsize)

    for i in range(n):
        ax[i].stem(X[i, :])
        ax[i].grid()

    plt.suptitle(title, fontsize=title_fontsize)
    plt.draw()


def plot_convergence_plot(
    metric,
    xlabel="",
    ylabel="",
    title="",
    figsize=(12, 8),
    fontsize=15,
    linewidth=3,
    colorcode="#050C12",
):

    plt.figure(figsize=figsize)
    plt.plot(metric, linewidth=linewidth, color=colorcode)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    # plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
    plt.grid()
    plt.draw()


def Subplot_gray_images(I, image_shape=[512, 512], height=15, width=15, title=""):
    n_images = I.shape[1]
    fig, ax = plt.subplots(1, n_images)
    fig.suptitle(title)
    fig.set_figheight(height)
    fig.set_figwidth(width)
    for i in range(n_images):
        ax[i].imshow(I[:, i].reshape(image_shape[0], image_shape[1]), cmap="gray")

    plt.show()


def Subplot_RGB_images(I, imsize=[3240, 4320], height=15, width=15, title=""):
    n_images = I.shape[0]
    Im = [I[i, :].reshape(imsize[0], imsize[1], 3) for i in range(I.shape[0])]
    fig, ax = plt.subplots(1, n_images, figsize=(25, 50))
    fig.suptitle(title)
    fig.set_figheight(height)
    fig.set_figwidth(width)
    for i in range(n_images):
        ax[i].imshow(Im[i])
        ax[i].axes.xaxis.set_visible(False)
        ax[i].axes.yaxis.set_visible(False)
    plt.subplots_adjust(
        right=0.97, left=0.03, bottom=0.03, top=0.97, wspace=0.1, hspace=0.1
    )
    plt.draw()

def SetPlotRC():
    #If fonttype = 1 doesn't work with LaTeX, try fonttype 42.
    plt.rc('pdf',fonttype = 42)
    plt.rc('ps',fonttype = 42)

def ApplyFont(ax, xlabel_text_size = 25.0, ylabel_text_size = 25.0, title_text_size = 19.0, ticks_text_size = 20):

    ticks = ax.get_xticklabels() + ax.get_yticklabels()
    text_size = ticks_text_size
    
    for t in ticks:
        t.set_fontname('DejaVu Sans')
        t.set_fontsize(text_size)

    txt = ax.get_xlabel()
    txt_obj = ax.set_xlabel(txt)
    txt_obj.set_fontname('DejaVu Sans')
    txt_obj.set_fontsize(xlabel_text_size)

    txt = ax.get_ylabel()
    txt_obj = ax.set_ylabel(txt)
    txt_obj.set_fontname('DejaVu Sans')
    txt_obj.set_fontsize(ylabel_text_size)

    # txt = ax.get_xticks()
    # txt_obj = txt
    # # txt_obj.set_fontname('DejaVu Sans')
    # txt_obj.set_fontsize(x_ticks_text_size)
    
    # txt = ax.get_yticks()
    # txt_obj.set_fontname('DejaVu Sans')
    # txt_obj.set_fontsize(yticks_text_size)
    
    
    txt = ax.get_title()
    txt_obj = ax.set_title(txt)
    txt_obj.set_fontname('DejaVu Sans')
    txt_obj.set_fontsize(title_text_size)
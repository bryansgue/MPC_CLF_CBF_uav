import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
plt.rc('text', usetex = True)
def fancy_plots_2():
    # Define parameters fancy plot
    pts_per_inch = 72.27
    # write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
    text_width_in_pts = 300.0
    # inside a figure environment in latex, the result will be on the
    # dvi/pdf next to the figure. See url above.
    text_width_in_inches = text_width_in_pts / pts_per_inch
    # make rectangles with a nice proportion
    golden_ratio = 0.618
    # figure.png or figure.eps will be intentionally larger, because it is prettier
    inverse_latex_scale = 2
    # when compiling latex code, use
    # \includegraphics[scale=(1/inverse_latex_scale)]{figure}
    # we want the figure to occupy 2/3 (for example) of the text width
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    # always 1.0 on the first argument
    fig_size = (1.0 * csize, 0.7 * csize)
    # find out the fontsize of your latex text, and put it here
    text_size = inverse_latex_scale * 10
    label_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8

    params = {'backend': 'ps',
            'axes.labelsize': text_size,
            'legend.fontsize': tick_size,
            'legend.handlelength': 2.5,
            'legend.borderaxespad': 0,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'font.family': 'serif',
            'font.size': text_size,
            # Times, Palatino, New Century Schoolbook,
            # Bookman, Computer Modern Roman
            # 'font.serif': ['Times'],
            'ps.usedistiller': 'xpdf',
            'text.usetex': True,
            'figure.figsize': fig_size,
            # include here any neede package for latex
            'text.latex.preamble': [r'\usepackage{amsmath}',
                ],
                }
    plt.rc(params)
    plt.clf()
    # figsize accepts only inches.
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13,
                        hspace=0.05, wspace=0.02)
    plt.ioff()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    return fig, ax1, ax2


def fancy_plot():
    # Define parameters fancy plot
    pts_per_inch = 72.27
    # write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
    text_width_in_pts = 300.0
    # inside a figure environment in latex, the result will be on the
    # dvi/pdf next to the figure. See url above.
    text_width_in_inches = text_width_in_pts / pts_per_inch
    # make rectangles with a nice proportion
    golden_ratio = 0.618
    # figure.png or figure.eps will be intentionally larger, because it is prettier
    inverse_latex_scale = 2
    # when compiling latex code, use
    # \includegraphics[scale=(1/inverse_latex_scale)]{figure}
    # we want the figure to occupy 2/3 (for example) of the text width
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    # always 1.0 on the first argument
    fig_size = (1.0 * csize, 0.7 * csize)
    # find out the fontsize of your latex text, and put it here
    text_size = inverse_latex_scale * 10
    label_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8

    params = {'backend': 'ps',
            'axes.labelsize': text_size,
            'legend.fontsize': tick_size,
            'legend.handlelength': 2.5,
            'legend.borderaxespad': 0,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'font.family': 'serif',
            'font.size': text_size,
            # Times, Palatino, New Century Schoolbook,
            # Bookman, Computer Modern Roman
            # 'font.serif': ['Times'],
            'ps.usedistiller': 'xpdf',
            'text.usetex': True,
            'figure.figsize': fig_size,
            # include here any neede package for latex
            'text.latex.preamble': [r'\usepackage{amsmath}',
                ],
                }
    plt.rc(params)
    plt.clf()
    # figsize accepts only inches.
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13,
                        hspace=0.05, wspace=0.02)
    plt.ioff()
    ax1 = fig.add_subplot(111)
    return fig, ax1

def plot_pose(x, xref, t):
    fig, ax = fancy_plot()
    
    colors = ['#BB5651', '#69BB51', '#5189BB', '#FFD700']  # Add color for psi
    labels = [r'$x$', r'$y$', r'$z$', r'$\psi$']
    
    for i in range(4):
        ax.plot(t[0:x.shape[1]], x[i, :],
                color=colors[i], lw=2, ls="-", label=labels[i])
        
        ax.plot(t[0:x.shape[1]], xref[i, 0:x.shape[1]],
                color=colors[i], lw=2, ls="--", label=labels[i] + r'$d$')

    ax.set_ylabel(r"$[states]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig

def plot_error(error, t):
    fig, ax = fancy_plot()
    
    colors = ['#BB5651', '#69BB51', '#5189BB', '#FFD700']  # Add color for psi
    labels = [r'$x$', r'$y$', r'$z$', r'$\psi$']
    
    for i in range(3):
        ax.plot(t[0:error.shape[1]], error[i, :],
                color=colors[i], lw=2, ls="-", label=labels[i])

    ax.set_ylabel(r"$[states]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig

def plot_time(ts, delta_t, t):
    fig, ax = fancy_plot()
    ax.set_xlim((t[0], t[-1]))
    
    colors = ['#BB5651', '#69BB51', '#5189BB', '#FFD700']  # Add color for psi
    labels = [r'$x$', r'$y$', r'$z$', r'$\psi$']
    
    for i in range(1):
        ax.plot(t[0:ts.shape[1]], ts[i, :],
                color='#BB5651', lw=2, ls="--", label=labels[i])
        
        ax.plot(t[0:ts.shape[1]], delta_t[i, 0:ts.shape[1]],
                color='#69BB51', lw=2, ls="-", label=labels[i] + r'$d$')

    ax.set_ylabel(r"$[states]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig

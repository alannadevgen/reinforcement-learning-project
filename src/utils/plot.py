import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    plt.title("Training")
    plt.xlabel('# games')
    plt.ylabel('Score')

    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean score")
    plt.ylim(ymin=0)
    
    plt.legend(framon=False)

    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
import numpy as np
import matplotlib.pyplot as plt
import ternary
from causal_learning import dag
from causal_learning import utils
from causal_learning.graph_active_learner import GraphActiveLearner
from causal_learning.graph_self_teacher import GraphSelfTeacher

if __name__ == "__main__":
    t = 0.8  # transmission rate
    b = 0.0  # background rate

    # get predictions of information gain model for all 27 problems
    active_learning_problems = utils.create_active_learning_hyp_space(t=t, b=b)
    ig_model_predictions = []
    tau = 0.37

    for i, active_learning_problem in enumerate(active_learning_problems):
        gal = GraphActiveLearner(active_learning_problem)
        gal.update_posterior()
        eig = gal.expected_information_gain().tolist()
        ig_model_predictions.append(eig)

    # get predictions of self-teaching model for all 27 problems
    active_learning_problems = utils.create_active_learning_hyp_space(t=t, b=b)
    self_teaching_model_predictions = []

    for i, active_learning_problem in enumerate(active_learning_problems):
        gst = GraphSelfTeacher(active_learning_problem)
        gst.update_learner_posterior()
        self_teaching_posterior = gst.update_self_teaching_posterior()
        self_teaching_model_predictions.append(self_teaching_posterior)

    figure, ax = plt.subplots()
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])

    for i in range(len(ig_model_predictions)):
        # make ternary plot
        scale = 1
        ax = figure.add_subplot(3, 9, i+1)
        tax = ternary.TernaryAxesSubplot(ax=ax)
        figure.set_size_inches(10, 10)
        tax.set_title("Problem {}".format(i+1), fontsize=10)
        tax.boundary(linewidth=2.0)
        tax.scatter([ig_model_predictions[i]], marker='o',
                    color='blue', label="Active Learning")
        tax.scatter([self_teaching_model_predictions[i]],
                    marker='o', color='red', label="Self-Teaching")
        tax.clear_matplotlib_ticks()
        ax.set_frame_on(False)
        handles, labels = ax.get_legend_handles_labels()

    figure.suptitle(
        "Comparing predictions from the Information Gain (blue) and Self-Teaching (red) models")
    figure.legend(handles, labels, loc='lower center')
    plt.show()

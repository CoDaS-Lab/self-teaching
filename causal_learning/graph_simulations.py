import numpy as np
import matplotlib.pyplot as plt
import ternary
from causal_learning import dag
from causal_learning import utils
from causal_learning.graph_active_learner import GraphActiveLearner
from causal_learning.graph_self_teacher import GraphSelfTeacher
from causal_learning.graph_positive_test_strategy import GraphPositiveTestStrategy

if __name__ == "__main__":
    t = 0.8  # transmission rate
    b = 0.0  # background rate

    active_learning_problems = utils.create_active_learning_hyp_space(t=t, b=b)
    ig_model_predictions = []
    self_teaching_model_predictions = []
    pts_model_predictions = []

    # get predictions of all three models
    for i, active_learning_problem in enumerate(active_learning_problems):
        gal = GraphActiveLearner(active_learning_problem)
        gal.update_posterior()
        eig = gal.expected_information_gain().tolist()
        ig_model_predictions.append(eig)

        gst = GraphSelfTeacher(active_learning_problem)
        gst.update_learner_posterior()
        self_teaching_posterior = gst.update_self_teaching_posterior()
        self_teaching_model_predictions.append(self_teaching_posterior)

        gpts = GraphPositiveTestStrategy(active_learning_problem)
        pts_model_predictions.append(gpts.positive_test_strategy())

    figure, ax = plt.subplots()
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])

    points_one = [(0.33, 0.33, 0.33), (0.5, 0.5, 0)]
    points_two = [(0.33, 0.33, 0.33), (0.5, 0, 0.5)]
    points_three = [(0.33, 0.33, 0.33), (0, 0.5, 0.5)]

    for i in range(len(ig_model_predictions)):
        # make ternary plot
        scale = 1
        ax = figure.add_subplot(3, 9, i+1)
        tax = ternary.TernaryAxesSubplot(ax=ax)
        figure.set_size_inches(10, 10)
        tax.set_title("Problem {}".format(i+1), fontsize=10)
        tax.boundary(linewidth=2.0)
        tax.scatter([ig_model_predictions[i]], marker='o',
                    color='red', label="Information Gain", alpha=0.6, s=80)
        tax.scatter([pts_model_predictions[i]],
                    marker='o', color='green', label="Positive-Test Strategy",
                    alpha=0.6, s=80)
        tax.scatter([self_teaching_model_predictions[i]],
                    marker='o', color='blue', label="Self-Teaching",
                    alpha=0.6, s=80)

        tax.line(points_one[0], points_one[1], color='black', linestyle=':')
        tax.line(points_two[0], points_two[1], color='black', linestyle=':')
        tax.line(points_three[0], points_three[1],
                 color='black', linestyle=':')
        tax.clear_matplotlib_ticks()
        ax.set_frame_on(False)
        handles, labels = ax.get_legend_handles_labels()

    figure.suptitle(
        "Comparing predictions from the Information Gain (red), Self-Teaching (blue) and Positive-Test Strategy (green) models")
    figure.legend(handles, labels, loc='lower center')
    plt.show()

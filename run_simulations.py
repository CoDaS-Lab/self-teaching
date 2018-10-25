import numpy as np
import matplotlib.pyplot as plt
import ternary
from models import utils
from models.concept_active_learner import ConceptActiveLearner
from models.concept_self_teacher import ConceptSelfTeacher
from models.graph_active_learner import GraphActiveLearner
from models.graph_self_teacher import GraphSelfTeacher
from models.graph_positive_test_strategy import GraphPositiveTestStrategy


def run_first_feature_boundary_simulations():
    hyp_space_type = "boundary"
    n_features = 3
    sampling = "max"

    figure, ax = plt.subplots()

    al = ConceptActiveLearner(n_features, hyp_space_type, sampling)
    active_learning_prob_one = np.array([
        al.expected_information_gain(x) for x in range(n_features)])

    # normalize
    active_learning_prob_one = active_learning_prob_one / \
        np.sum(active_learning_prob_one)

    # get predictions from self-teaching model
    st = ConceptSelfTeacher(n_features, hyp_space_type, sampling)
    st.update_learner_posterior()
    st.update_self_teaching_posterior()
    self_teacher_prob_one = st.self_teaching_posterior[0, :, 0]

    plt.figure()
    plt.plot(np.arange(1, n_features + 1), active_learning_prob_one,
             color='red',
             # linestyle='',
             label="Information Gain model")
    plt.plot(np.arange(1, n_features + 1), self_teacher_prob_one,
             color='blue',
             label="Self-teaching model")
    axes = plt.gca()
    axes.set_ylim([0, 0.5])
    plt.xticks(np.arange(1, n_features+1), fontsize=16)
    plt.yticks(fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(loc='lower center', fontsize=16)
    plt.xlabel("Feature", fontsize=16)
    plt.ylabel("Probability of selecting feature", fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/concept_learning_boundary_first_feature_prob.png')


def run_eight_feature_boundary_simulations():
    hyp_space_type = "boundary"
    n_features = 8
    sampling = "max"

    figure, ax = plt.subplots()

    al = ConceptActiveLearner(n_features, hyp_space_type, sampling)
    active_learning_prob_one = np.array([
        al.expected_information_gain(x) for x in range(n_features)])

    # normalize
    active_learning_prob_one = active_learning_prob_one / \
        np.sum(active_learning_prob_one)

    # get predictions from self-teaching model
    st = ConceptSelfTeacher(n_features, hyp_space_type, sampling)
    st.update_learner_posterior()
    st.update_self_teaching_posterior()
    self_teacher_prob_one = st.self_teaching_posterior[0, :, 0]

    plt.figure()
    plt.plot(np.arange(1, n_features + 1), active_learning_prob_one,
             color='red',
             # linestyle=':',
             label="Information Gain model")
    plt.plot(np.arange(1, n_features + 1), self_teacher_prob_one,
             color='blue',
             label="Self-teaching model")
    axes = plt.gca()
    axes.set_ylim([0, 0.5])
    plt.xticks(np.arange(1, n_features+1), fontsize=16)
    plt.yticks(fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(loc='lower center', fontsize=16)
    plt.xlabel("Feature", fontsize=16)
    plt.ylabel("Probability of selecting feature", fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/concept_learning_boundary_eight_feature_prob.png')


def run_second_feature_boundary_simulations():
    hyp_space_type = "boundary"
    n_hyp = 4
    n_features = 3
    n_labels = 2
    sampling = "max"

    # feature, label pairs
    xs = [0, 1, 1, 2]
    ys = [1, 0, 1, 0]

    figure, ax = plt.subplots()
    figure.set_size_inches(10, 10)

    for i, (x, y) in enumerate(zip(xs, ys)):
        # get predictions from active learning model
        al = ConceptActiveLearner(n_features, hyp_space_type, sampling)
        active_learning_prob_one = np.array([
            al.expected_information_gain(x) for x in range(n_features)])

        # normalize
        active_learning_prob_one = active_learning_prob_one / \
            np.sum(active_learning_prob_one)

        # perform update
        al.update(x=x, y=y)
        active_learning_prob_two = np.array([
            al.expected_information_gain(x) for x in range(n_features)])

        # normalize
        denom = np.sum(active_learning_prob_two)
        active_learning_prob_two = np.divide(active_learning_prob_two, denom,
                                             where=denom != 0)
        active_learning_prob_two[np.isclose(denom, 0)] = 0

        # get predictions from self-teaching model
        st = ConceptSelfTeacher(n_features, hyp_space_type, sampling)
        st.update_learner_posterior()
        st.update_self_teaching_posterior()
        self_teacher_prob_one = st.self_teaching_posterior[0, :, 0]

        # update learner posterior after a single observation
        updated_learner_posterior = st.learner_posterior[:, x, y]
        st.learner_posterior = np.repeat(
            updated_learner_posterior,
            n_labels * n_features).reshape(
                n_hyp, n_features, n_labels)

        # update learner and self-teacher after a single observation
        st.update_learner_posterior()
        st.update_self_teaching_posterior()
        self_teacher_prob_two = st.self_teaching_posterior[0, :, 0]

        # plot second feature prob
        plt.subplot(2, 2, i+1)
        plt.plot(np.arange(1, n_features + 1), active_learning_prob_two,
                 color='red',
                 # linestyle=':',
                 label="Information Gain model")
        plt.plot(np.arange(1, n_features + 1), self_teacher_prob_two,
                 color='blue',
                 label="Self-teaching model")
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.xlabel("Feature", fontsize=16)
        plt.ylabel("Probability of selecting feature", fontsize=16)
        plt.title('x = {}, y = {}'.format(x+1, y), fontsize=18)
        plt.xticks([1, 2, 3], fontsize=16)
        plt.yticks(fontsize=16)
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig('figures/concept_learning_boundary_second_feature_prob.png')


def run_three_feature_line_simulations():
    hyp_space_type = "line"
    n_hyp = 6
    n_features = 3
    n_labels = 2
    sampling = "max"

    # feature, label pairs
    xs = [0, 0, 1, 1, 2, 2]
    ys = [0, 1, 0, 1, 0, 1]

    figure, ax = plt.subplots()

    al = ConceptActiveLearner(n_features, hyp_space_type, sampling)
    active_learning_prob_one = np.array([
        al.expected_information_gain(x) for x in range(n_features)])

    # normalize
    active_learning_prob_one = active_learning_prob_one / \
        np.sum(active_learning_prob_one)

    # get predictions from self-teaching model
    st = ConceptSelfTeacher(n_features, hyp_space_type, sampling)
    st.update_learner_posterior()
    st.update_self_teaching_posterior()
    self_teacher_prob_one = st.self_teaching_posterior[0, :, 0]

    plt.figure()
    plt.plot(np.arange(1, n_features + 1), active_learning_prob_one,
             color='red',
             # linestyle=':',
             label="Information Gain model")
    plt.plot(np.arange(1, n_features + 1), self_teacher_prob_one,
             color='blue',
             label="Self-teaching model")
    axes = plt.gca()
    # axes.set_ylim([0.1, 0.15])
    plt.xticks(np.arange(1, n_features+1), fontsize=16)
    plt.yticks(fontsize=16)
    axes.set_ylim([0, 0.4])
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(loc='lower center', fontsize=16)
    plt.xlabel("Feature", fontsize=16)
    plt.ylabel("Probability of selecting feature", fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/concept_learning_line_three_features.png')


def run_eight_feature_line_simulations():
    hyp_space_type = "line"
    n_hyp = 36
    n_features = 8
    sampling = "max"

    figure, ax = plt.subplots()

    al = ConceptActiveLearner(n_features, hyp_space_type, sampling)
    active_learning_prob_one = np.array([
        al.expected_information_gain(x) for x in range(n_features)])

    # normalize
    active_learning_prob_one = active_learning_prob_one / \
        np.sum(active_learning_prob_one)

    # get predictions from self-teaching model
    st = ConceptSelfTeacher(n_features, hyp_space_type, sampling)
    st.update_learner_posterior()
    st.update_self_teaching_posterior()
    self_teacher_prob_one = st.self_teaching_posterior[0, :, 0]

    plt.figure()
    plt.plot(np.arange(1, n_features + 1), active_learning_prob_one,
             color='red',
             # linestyle=':',
             label="Information Gain model")
    plt.plot(np.arange(1, n_features + 1), self_teacher_prob_one,
             color='blue',
             label="Self-teaching model")
    axes = plt.gca()
    axes.set_ylim([0, 0.2])
    plt.xticks(np.arange(1, n_features + 1), fontsize=16)
    plt.yticks(fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(loc='lower center', fontsize=16)
    plt.xlabel("Feature", fontsize=16)
    plt.ylabel("Probability of selecting feature", fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/concept_learning_line_eight_features.png')


def run_causal_subset_simulations():
    t = 0.8  # transmission rate
    b = 0.0  # background rate

    active_learning_problems = utils.create_active_learning_hyp_space(t=t, b=b)
    active_learning_subset = [active_learning_problems[i] for i in [7, 13, 15]]
    ig_model_predictions = []
    self_teaching_model_predictions = []

    # get predictions of all three models
    for i, active_learning_problem in enumerate(active_learning_subset):
        # gal = GraphActiveLearner(active_learning_problem)
        # gal.update_posterior()
        # eig = gal.expected_information_gain().tolist()
        # ig_model_predictions.append(eig)

        gst = GraphSelfTeacher(active_learning_problem)
        gst.update_learner_posterior()
        self_teaching_posterior = gst.update_self_teaching_posterior()
        self_teaching_model_predictions.append(self_teaching_posterior)

    return ig_model_predictions, self_teaching_model_predictions


def run_causal_simulations():
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
    figure.set_size_inches(16, 5)

    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])

    points_one = [(0.33, 0.33, 0.33), (0.5, 0.5, 0)]
    points_two = [(0.33, 0.33, 0.33), (0.5, 0, 0.5)]
    points_three = [(0.33, 0.33, 0.33), (0, 0.5, 0.5)]

    for i in range(len(ig_model_predictions)):
        # make ternary plot
        ax = figure.add_subplot(3, 9, i+1)
        tax = ternary.TernaryAxesSubplot(ax=ax)
        tax.set_title("Problem {}".format(i+1), fontsize=10)
        tax.boundary(linewidth=2.0)
        tax.scatter([ig_model_predictions[i]], marker='o',
                    color='red', label="Information Gain", alpha=0.8, s=40)
        tax.scatter([pts_model_predictions[i]],
                    marker='d', color='green', label="Positive-Test Strategy",
                    alpha=0.8, s=40)
        tax.scatter([self_teaching_model_predictions[i]],
                    marker='s', color='blue', label="Self-Teaching",
                    alpha=0.8, s=40)

        tax.line(points_one[0], points_one[1],
                 color='black', linestyle=':')
        tax.line(points_two[0], points_two[1],
                 color='black', linestyle=':')
        tax.line(points_three[0], points_three[1],
                 color='black', linestyle=':')
        tax.left_axis_label("x_1", fontsize=20)
        tax.right_axis_label("x_2", fontsize=20)
        tax.bottom_axis_label("x_3", fontsize=20)

        tax.clear_matplotlib_ticks()
        ax.set_frame_on(False)
        handles, labels = ax.get_legend_handles_labels()

    plt.savefig('figures/causal_learning_simulations.png', dpi=600)


if __name__ == "__main__":
    run_first_feature_boundary_simulations()
    run_eight_feature_boundary_simulations()
    run_second_feature_boundary_simulations()
    run_three_feature_line_simulations()
    run_eight_feature_line_simulations()
    run_causal_simulations()

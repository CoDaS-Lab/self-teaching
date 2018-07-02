import numpy as np
import matplotlib.pyplot as plt
from concept_learning.active_learner import ActiveLearner
from concept_learning.self_teacher import SelfTeacher

if __name__ == "__main__":
    hyp_space_type = "boundary"
    n_hyp = 4
    n_features = 3
    n_labels = 2
    sampling = "max"

    # feature, label pairs
    xs = [0, 0, 1, 1, 2, 2]
    ys = [0, 1, 0, 1, 0, 1]

    figure, ax = plt.subplots()

    for i, (x, y) in enumerate(zip(xs, ys)):
        # get predictions from active learning model
        al = ActiveLearner(n_features, hyp_space_type, sampling)
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
        st = SelfTeacher(n_features, hyp_space_type, sampling)
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

        plt.subplot(2, 3, i+1)
        # plt.plot(np.arange(1, 4), active_learning_prob_one,
        #          color='green',
        #          label="Active learner first feature prob")
        # plt.plot(np.arange(1, 4), self_teacher_prob_one,
        #          color='orange',
        #          label="Self-teacher first feature prob")
        plt.plot(np.arange(3), active_learning_prob_two,
                 color='red',
                 label="Active learner second feature prob")
        plt.plot(np.arange(3), self_teacher_prob_two,
                 color='blue',
                 label="Self-teacher second feature prob")
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.title('x = {}, y = {}'.format(x, y))
        plt.xticks([0, 1, 2])
        handles, labels = ax.get_legend_handles_labels()

    plt.suptitle(
        "Comparing predictions from the Active Learning (red) and Self-Teaching (blue) models\nfor selecting the second feature in the concept learning game")
    figure.legend(handles, labels, loc='lower center')
    plt.show()

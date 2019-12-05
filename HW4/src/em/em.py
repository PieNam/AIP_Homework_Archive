import numpy
import scipy.stats as ss

observations = numpy.array([[1,0,0,0,1,1,0,1,0,1], [1,1,1,1,0,1,1,1,1,1],
                            [1,0,1,1,1,1,1,0,1,1], [1,0,1,0,0,0,1,1,0,0],
                            [0,1,1,1,0,1,1,1,0,1]])

p_a = 0.5
p_b = 0.5

def em_step(p_a, p_b, observations):
    a_h = a_t = b_h = b_t = 0

    theta_a = p_a
    theta_b = p_b

    # E step
    for obs in observations:
        obs_len = len(obs)
        total_head = obs.sum()
        total_tail = obs_len - total_head
        contribution_a = ss.binom.pmf(total_head, obs_len, theta_a)
        contribution_b = ss.binom.pmf(total_head, obs_len, theta_b)
        weight_a = contribution_a / (contribution_a + contribution_b)
        weight_b = contribution_b / (contribution_a + contribution_b)
        a_h += weight_a * total_head
        a_t += weight_a * total_tail
        b_h += weight_b * total_head
        b_t += weight_b * total_tail

    # M step
    p_a_t = a_h / (a_h + a_t)
    p_b_t = b_h / (b_h + b_t)

    return p_a_t, p_b_t

def em(observations, p_a, p_b, threshold, iterations):
    iteration_count = 0
    while iteration_count < iterations:
        iteration_count += 1
        print("iteration [", iteration_count, "]: P(A) =", p_a, ", P(B) =", p_b)
        p_a_t, p_b_t = em_step(p_a, p_b, observations)
        delta = numpy.abs(p_a_t - p_a)
        if delta < threshold:
            break
        else:
            p_a = p_a_t
            p_b = p_b_t
    
    return p_a, p_b, iteration_count

if __name__ == "__main__":
    situation_a_theta_a = (observations[1].sum() + observations[2].sum() + observations[4].sum()) / 30
    situation_a_theta_b = (observations[0].sum() + observations[3].sum()) / 20
    p_a, p_b, it = em(observations, 0.5, 0.2, 1e-6, 10000)
    print("Result of situation (b): P(A) =", p_a, ", P(B) =", p_b, ". after", it, "iterations.")
    print("Result of situation (a): P(A) =", situation_a_theta_a, ", P(B) =", situation_a_theta_b, ".")

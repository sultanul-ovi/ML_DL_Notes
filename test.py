import numpy as np
import scipy.stats as stats

# Data for Algorithms A and B
algorithm_a_scores = np.array([141, 181, 200, 187, 169, 173, 186, 195, 177, 182, 133, 183, 197, 165, 180, 198])
algorithm_b_scores = np.array([175, 164, 172, 194, 176, 197, 154, 134, 168, 164, 185, 159, 161, 189, 170, 164])

# Combine and rank the data
combined_scores = np.concatenate([algorithm_a_scores, algorithm_b_scores])
ranks = stats.rankdata(combined_scores)

# Calculate rank sums
rank_sum_a = np.sum(ranks[:len(algorithm_a_scores)])
rank_sum_b = np.sum(ranks[len(algorithm_a_scores):])

# Calculate U statistics for both algorithms
n1 = len(algorithm_a_scores)
n2 = len(algorithm_b_scores)
u1 = rank_sum_a - (n1 * (n1 + 1) / 2)
u2 = rank_sum_b - (n2 * (n2 + 1) / 2)

# Calculate mean (mu) and standard deviation (sigma) for U statistics
mu = n1 * n2 / 2
sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

# Calculate Z-scores for both U1 and U2
z_score_u1 = (u1 - mu) / sigma
z_score_u2 = (u2 - mu) / sigma

# # Output the results
print("U statistic for Algorithm A (U1):", u1)
print("U statistic for Algorithm B (U2):", u2)
print("Z-score for U1:", z_score_u1)
print("Z-score for U2:", z_score_u2)

# Critical values
alpha = 0.05
critical_z_value = stats.norm.ppf(1 - alpha)  # For one-tailed test
critical_u_value = 75

# Test the hypothesis using U1, U2, and Z-scores
conclusion_u1 = "Reject H0" if u1 < critical_u_value else "Fail to reject H0"
conclusion_u2 = "Reject H0" if u2 < critical_u_value else "Fail to reject H0"
conclusion_z1 = "Reject H0" if z_score_u1 > critical_z_value else "Fail to reject H0"
conclusion_z2 = "Reject H0" if z_score_u2 > critical_z_value else "Fail to reject H0"


print("u1", conclusion_u1)
print("u2", conclusion_u2)
print("z1", conclusion_z1)
print("z2", conclusion_z2)

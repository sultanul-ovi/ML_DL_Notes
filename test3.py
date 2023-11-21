import numpy as np
import scipy.stats as stats

# Replace these with your actual data
algorithm_a_scores = np.array([141, 181, 200, 187, 169, 173, 186, 195, 177, 182, 133, 183, 197, 165, 180, 198])
algorithm_b_scores = np.array([175, 164, 172, 194, 176, 197, 154, 134, 168, 164, 185, 159, 161, 189, 170, 164])

# Combine and rank the data
combined_scores = np.concatenate([algorithm_a_scores, algorithm_b_scores])
ranks = stats.rankdata(combined_scores)


# Calculate rank sums
rank_sum_a = np.sum(ranks[:len(algorithm_a_scores)])
rank_sum_b = np.sum(ranks[len(algorithm_a_scores):])

# Calculate U statistics
n1, n2 = len(algorithm_a_scores), len(algorithm_b_scores)
u1 = rank_sum_a - (n1 * (n1 + 1) / 2)
u2 = rank_sum_b - (n2 * (n2 + 1) / 2)
smaller_u = min(u1, u2)

# Calculate mean and standard deviation for U distribution
mu = n1 * n2 / 2
sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

# Calculate Z-score
z_score = (smaller_u - mu) / sigma

# Determine the critical Z-value (for a one-tailed test at α = 0.05)
critical_z_value = stats.norm.ppf(1 - 0.05)

# Make a decision
if z_score > critical_z_value:
    conclusion = "Reject H0"
else:
    conclusion = "Fail to reject H0"

# Output the results
print("Z-score:", z_score)
print("Conclusion:", conclusion)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Iris.csv")

if "Id" in df.columns:
    df = df.drop(columns=["Id"])

feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = df[feature_cols].to_numpy(dtype=float)
y = df["Species"].to_numpy()

#Z-score normalization
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X.mean(axis=0)) / X.std(axis=0)

def scatter_plot(df, feature_cols, centroids=None, mode = "before", new_point=None):
    species_list = df["Species"].unique()
    colors = {"Iris-setosa":"red", "Iris-versicolor":"green", "Iris-virginica":"blue"}

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axes = axes.flatten()
    plot_index = 0

    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            ax = axes[plot_index]
            if mode == "before":
                ax.scatter(df[feature_cols[i]], df[feature_cols[j]], c="gray", alpha=0.6, label="Data")
            else:
                for sp in species_list:
                    subset = df[df["Species"] == sp]
                    ax.scatter(subset[feature_cols[i]], subset[feature_cols[j]],
                               label=sp, c=colors[sp], alpha=1)

                if centroids is not None:
                    ax.scatter(centroids[:, i], centroids[:, j],marker="X", s=200, c="black", label="Centroid")
                if new_point is not None:
                    ax.scatter([new_point[i]], [new_point[j]],marker="o", s=120, c="black", label="New point", edgecolors="white", linewidths=0.8)
            ax.set_xlabel(feature_cols[i])
            ax.set_ylabel(feature_cols[j])
            ax.set_title(f"{feature_cols[i]} vs {feature_cols[j]} (theo Species)")
            plot_index += 1


    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')
    plt.tight_layout()
    plt.show()

## X: 10 hang 2 cot nghia la 10 diem 2 cum
# Calculate sum_square_dist
def sum_sq_dists(X, C):
    X2 = np.sum(X*X, axis=1).reshape(-1,1)
    C2 = np.sum(C*C, axis=1).reshape(1,-1)
    XC = np.dot(X, C.T)
    return X2 + C2 - 2 * XC

#Find the nearest centroid
def assign_labels_sq(X, C):
    Dist = sum_sq_dists(X, C)
    return np.argmin(Dist, axis=1)

def update_centroid(X, labels, K, old_C=None, rng = None):
    rng = np.random.default_rng(0)
    new_C = np.zeros((K, X.shape[1]), dtype = X.dtype)
    for k in range(K):
        Xk = X[labels == k]
        if Xk.size > 0:
            new_C[k] = Xk.mean(axis = 0) #tim vi tri tam cum
        else:
            if old_C is not None:
                new_C = old_C
            else:
                randomX = rng.integers(0, X.shape(0))
                new_C[k] = X[randomX]
    return new_C

def fit_sq(X, K, max_iter=300, tol=1e-6):
    rng = np.random.default_rng(0)
    C = X[rng.choice(X.shape[0], size=K, replace=False)].copy()
    for _ in range(max_iter):
        labels = assign_labels_sq(X, C)
        C_new = update_centroid(X, labels, K, old_C=C, rng=rng)
        norm2 = np.sqrt(np.sum((C_new-C)*(C_new-C)))
        if norm2 < tol:
            C = C_new
            break
        C = C_new

    D2 = sum_sq_dists(X, C)
    inertia = float(np.sum(np.min(D2, axis=1)))
    return C, labels, inertia

def optimal_K():
    K_values = range(1, 10)
    inertias = []

    for K in K_values:
        _, _, inert = fit_sq(X, K, max_iter=300, tol=1e-6)
        inertias.append(inert)
        print(f"K={K}, inertia={inert:.3f}")


    reduction_rates = []
    for i in range(1, len(inertias)):
        reduction_rate = (inertias[i - 1] - inertias[i]) / inertias[i - 1]
        reduction_rates.append(reduction_rate)
        print(f"K{i} -> K{i+1}: {reduction_rate:.4f}")

    optimal_k = 0
    max_diff = 0
    th = 0.1
    for i in range(1, len(reduction_rates)):
        diff = reduction_rates[i - 1] - reduction_rates[i]
        if  diff > max_diff and diff > th:
            max_diff = diff
            print('\n')
            print(reduction_rates[i - 1] - reduction_rates[i])
            optimal_k = i + 2
            break

    print(f"\nK tối ưu theo Elbow method: {optimal_k}")
    return K_values, inertias, optimal_k

def plot_inertias(K_values, inertias):
    plt.figure(figsize=(7,5))
    plt.plot(list(K_values), inertias, marker='o')
    plt.xlabel("Số cụm K")
    plt.ylabel("Inertia (tổng bình phương khoảng cách)")
    plt.title("Elbow method (chọn K cho K-Means)")
    plt.grid(True)
    plt.show()

def predict_cluster(user_input, C_final, X_mean, X_std):
    x = np.array(user_input, dtype=float).reshape(1, -1)
    X = (x - X_mean) / X_std
    distances = sum_sq_dists(X, (C_final - X_mean) / X_std)
    cluster = np.argmin(distances, axis=1)[0]
    return cluster

def user_input(C_final_orig):
    print("\n--- Nhập dữ liệu mới để phân cụm ---")
    sepal_len = float(input("Sepal Length (cm): "))
    sepal_wid = float(input("Sepal Width (cm): "))
    petal_len = float(input("Petal Length (cm): "))
    petal_wid = float(input("Petal Width (cm): "))
    new_point = [sepal_len, sepal_wid, petal_len, petal_wid]
    cluster = predict_cluster(new_point, C_final_orig, X_mean, X_std)
    print(f"Điểm {new_point} thuộc về cụm số: {cluster + 1}")
    return np.array(new_point, dtype=float)

def main():
    scatter_plot(df, feature_cols)
    K_values, inertias, optimal_k = optimal_K()
    plot_inertias(K_values, inertias)
    C_final, labels_final, inertia_final = fit_sq(X, optimal_k, max_iter=300, tol=1e-6)
    C_final_orig = C_final * X_std + X_mean     #chuan hoa --> cm
    print(f"Inertia(K_best) = {inertia_final:.3f}")
    print("\nCenters found by our algorithm (đơn vị cm, thứ tự feature):")
    print(pd.DataFrame(C_final_orig, columns=feature_cols))
    scatter_plot(df, feature_cols, C_final_orig, mode="colored")
    new_point = user_input(C_final_orig)
    scatter_plot(df, feature_cols, C_final_orig, mode="colored", new_point=new_point)
if __name__ == "__main__":
    main()
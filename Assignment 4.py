# Luisa Borriello Assignment 4

DIFFERENCE BETWEEN AI AND MACHINE LEARNING: Artificial intelligence is a research project, started to be conducted in 1946, during the Second World War, with military application, while Machine learning is one of the approaches that was taken in order to try to make human a machine, a computer. It is a part of artificial intelligence, which 
acquires information; has to see, has to hear, has to smell and then has to process the informations and decide what to do. We can define machine learning (chat GPT) with a simple relationship. Y_i=f(x_1,….x_k )+u_i. I want to predict, know y, based on K characteristics. u_i is ineliminable error, is the error that I make in predicting y, 
no matter how many x’s are used This function f is unknown, we have to estimate it, we have to learn it. “Estimating” in machine learning is to learn. This function in machine learning is called the machine, because you put the x’s into the machine, you put information, the machine elaborates these information and combine these information, and it gives you the y 
We have 2 kinds of data:
1) Data in which you have y, x_1,x_2, x_3. In this case we call the machine learning SUPERVISED LEARNING, because you’re learning the function f using x, using the supervision of y.
2) Sometimes we only have the x’s; we still want to say something about y, but we never get to see y. We only have characteristics. We’re still learning x saying something about y, but this is UNSUPERVISED, because we never get to see y, and still we have to learn the function f. 

GENERAL FRAMEWORK. If the variable y is continuous, the learning task is called “regression” task (task means I have to learn the function f using the x’s and y’s as a continuous function), while when y is discrete (predict if a person is employed or unemployed, or how many children a person has), we talk about “classification task”, 
because you think about being employed and unemployed as a class (you have the class of unemployed, the class of employed, and then you use the x’s to decide in which class you belong, you classify people).

The task of econometrics is to estimate the parameters (in econometrics we care about the causal effect, and we want to interpret what Beta is), while ML cares about predictions.

In time series we have data from 1 to t, we learn the model on this data, we estimate our model, we predict t+1, we want to say something about the future variable at t+1.  
The application of machine learning is not in time series, is in CROSS-SECTION. We have data, this dataset have supervision (is a supervised dataset), and this supervised dataset gives the information we want. Using these data I’m gonna try to learn f, I’m gonna try to learn information to make prediction of y, but then this f have hat 
prediction (is not the true f, is an estimated f), I use this estimated f on new data, where I have only the x’s. I have data on y_x, I’m gonna learn f, and I’m gonna apply this function f to new data, in order to predict y. This dataset in which you have both y and x is called the TRAINING DATASET; because using this dataset we’re gonna 
train the algorithm that is used to learn the function f. 

HOW CAN I EVALUATE HOW GOOD MY MODEL IS? Is very easy to make a prediction (you give me the x, I randomly give y, there will be huge errors between the true y and the y predicted by the model). In ML evaluating the model is very complicated (you have to apply to data that you haven’t seen). 
In econometrics you’re interested about Beta, and Beta is the causal effect (the effect of x on y), you don’t get to see the true effect, and so you have to think whether our model it is a good model or a bad model; you never get to see the true Beta, the true Beta is an idea.  Also here it’s almost the same, you never see the prediction,
you can get a very good idea of what the error using a model is going to be once you use it on data that you still don’t have. In machine learning you can estimate how well the model will perform on data that was not used to train the model

I use a dataset of Italian workers, I have 10 thousand Italian workers. I have a training sample; for this 10 thousand people I have wage, weekly working hours, region where this person is living, educational level, employer’s age, employer’s gender, whether you’re a white collar or a blue collar, temporary worker or full-time worker, whether you’re married or not, years of education

TRAINING AND VALIDATION STUFF. We split the dataset in 2 parts:
1) ACTUAL TRAINING SAMPLE (80% of my sample); 
2) VALIDATION (20% of my sample)

Once I learn the function f, I estimate f, I apply f to the x’s, I apply it to the validation set, I evaluate my model, I see what my model predicts and what the y of the model are; since I have  both the predictions and the y I can take the difference, and calculate the error of my model. I learn the model in the training sample, I apply outside in the validation set.
In the dataset of workers in 2019, I have only 1 x, x is age, y is wage. After set the randomness of the system, I run the function train test split; I have two x’s, one is the train, one is the validation, and 2 y’s, one is the train and one is the  validation, and I’m gonna do the x train and y train to train my model, and then I’m going to use my x validations to construct the predictions 
and then see how close the prediction are to the y, corresponding to the x validations.In the simple linear regression, we don’t care about β_1, the only thing we care is whether this linear combination of age is able to capture the wage of a person. β_1 is not a coefficient, is the weight that weights the wage variable. We have to minimize the sum of squared residuals.
wage=β_0+ β_1*age+e     y=Xβ+e

EVALUATING MODEL PERFORMANCE. Take the different between the actual y and the predicted y and square it, and take the average (this is called MEAN SQUARED ERROR). Root mean squared error is the square root of MSE 
1) IN SAMPLE, means I train my data, I fit my model using x trained and y trained, then I apply my model to x trained, and it gives me a prediction for y trained. MSE is very weak, univariate model makes an error of 497 euros on average, R squared is 0.03, means that 3% of variability in wages are captured by age
2) OUT OF SAMPLE, means use the trained data to get f, and then I apply to the validation set the function, I get the predicted y, I calculate the errors, the R^2. I apply the data out of sample, more or less I obtain the same results; the error is 485 (a little smaller), R^2 is 3%.

If we complicate the model, if we use MULTI VARIABLES, instead of using just the age we use other variables (education level, regression, sg level, married or not, male or female, age; we divide the variables between categorical and numerical). We standard scale the variable, we subtract the mean and we divide by the standard deviation. We estimate β, we apply our model to the new data,
our β’s are calculated for variables that have mean 0, and standard deviation 1, our new data are not gonna have mean 0 and standard deviation 1, so we should normalize them; we normalize them by taking the standard deviation and the mean for the training sample, and apply them to the validation sample. 
I’m gonna fit the models, I’m gonna make predictions of the trains, and I’m gonna make prediction for the validation, and I’m gonna predict how good my model is. RMSE in sample is 333.24 and R^2 in sample is 0.56 (56% of the variance is captured by the variables I put into my model), out of sample, RMSE is 331.34, R^2 out of sample is 0.55 (very similar; same results). 

NON-LINEARITY. Training data. {Y_i,x_(1i ),……..x_ki)
Linear model. Y_i= β_0+β_1 x_1i+…..+β_k x_ki + u_i
Non-linear model. β_0+β_1 x_1i+…….+β_k x_ki + β_(k+1) x_1i^2 +…..+β_2k x_ki^2 +…….

To introduce non-linearity in the model, we add polynomial features and interactions (the effect of a variable depends on another variable). In sample the model does better, while out of sample the model does worse (OVERFITTING). I have y and x, I have 2 observations, 2 people, there is only 1 x, but I have 2 datapoints, If I run an OLS, the OLS goes to the 2 points, R^2 is 1 
(I’m doing a perfect prediction of my model).I think that both of the guys are terrible, are lazy. When we add another guy, that does not fit (you made a big mistake, we have a COGNITIVE BIAS); you predict that should be there, so you make an error, you see another guy from Sicily that is not like the other one, the other 2, doesn’t smell bad, is not terrible
The way in which I can MITIGATE the error, in which I’m still using the 2 datapoints, still using the same model, instead of fitting the line that goes to the points, I can use a line that doesn’t minimize the sum of squared residuals, it does but only up to a certain point, you don’t trust the data that much, you trust it a little bit, not at a full extent.

One contribution of ML to statistics is REGULARIZATION; if the model is too complex, you can't rely on finding the best line, you can move your model in a way that a big β should be close to zero; instead of minimizing the sum of squared residuals, I’m gonna add a penalty (β is the reactivity of y to x’s, so it says “don’t give β too big”don’t make y too reactive to the x of this β). I try to flatten the model. 
More non-linear terms you’re gonna include, the more the function is gonna get closer to the true function f. Once I introduce polynomials and interactions, I increase the numbers of variables. The more non-linearity I include, the more parameters I have to estimate. From a statistical point of view, estimating more parameters, means losing more degrees of freedom. 
OVERVFITTING means that the number of features, the number of x’s that go into the model, the actual x’s used in my model is too large compared to the number of data that I have; I have a model that is too complex for the data Since the data is very sparse, the ratio between the number of observations and parameters is close to 1, that means that we really have lots of data. I’m fitting a model that is too complex,
I’m trusting too much the data and I’m trying to use this very complicated model in order to fit the data that I have. The solution is that instead of trusting the data, I’m gonna fit a line (this is not the best possible line, this does not minimize the sum of squared residuals), I’m gonna choose the parameter in order to almost minimize the sum of squared residuals; this means that the line is gonna be flatter, because the β is going to be smaller. 

LASSO AND RIDGE. In order to limit overfitting, in order to regularize the matrix, due to the fact that ML has billions of data points and not trivial parameters, was introduced the estimator called Ridge, that adds a ridge to the eigenvalues profile. The penalty is squared because you want positive and negative to be penalized in the same way. Instead of considering the square, you can consider the absolute value, that does the same things of the square in a different way.
〖min〗_(β_(0,…….) β_ ) ∑_(i=1)^n▒〖(Y_i 〗 - β_0 - β_1 x_1…..) ² + α ∑_(i=1)^n▒〖|β_j |〗

For large values of x, the penalty of the absolute value is smaller because the square is gonna be big, instead for values close to zero, the penalty is gonna be larger for the absolute value. The estimator with the absolute value is called Lasso (Least Absolute Shrinkage and Selection Operator), invented by Robert Tibshirani, a statistician in Stanford, that in 1980’s proposed to change the Ridge to the absolute value. Lasso solution is “sparse”; if you minimize the 
sum of squared residuals and you put a penalty term that is with the absolute value, some of the x's are gonna be equal to 0, some are different from zero, some are smaller than the one you would get with an OLS. In 1980’s one of the typical applications in ML was for DNA sequencing. There are 4 proteins, all with different sequences of DNA, you have to find all different combinations, that give rise to a regression with all different combinations, many of these β’s are gonna be equal to zero. 
The sum of the β is exactly equal to zero. Lasso is the perfect solution for this problem, in which you know ex ante that the sum of the true Beta of the f function is equal to zero (Lasso model selection).If alpha (degree of penalization) is equal to zero, I go back to the OLS, if alpha is equal to infinity, the β is zero. Alpha can be anything between zero and infinity. 
The first problem is the choice of alpha. Due to the fact that alpha cannot be estimated (we can estimate β given alpha), alpha is chosen by cross-validation. Cross-validation gives a good indication of how the model is going to perform out of sample. Given a model with 2 x’s, and I want to apply the Lasso. 
Y_i = β_0+β_1 x_1i+β_2 x_2i+u_i
〖min〗_(β_(0,) β_(1,)  β_2 )∑ (y_i-β_0-β_1 x_1i-β_2 x_2i)²+ α (β_1 ²+ β_2)²
We can estimate β_0,β_1,β_2, the size of β in a regression depends on y and on the scale of the x’s.  The constraint (β_1 ²+ β_2 ²) is sensitive to the scale of the data. With this constraint, we’re penalizing β’s that are large. In order for this constraint to make sense, all the x’s should be on the same scale. We normalize or standardize the variables (means we subtract the mean and divide by the standard deviation), so that the variables have all the same scale, have all mean 0 and standard deviation 1.

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting

plt.rcParams['figure.figsize'] = (10, 6)

# Set parameters for the data generating process
n_samples = 200        # Number of samples, number of observations
n_features = 50        # Total number of features
n_informative = 10      # Number of features with non-zero coefficients
noise_level = 1.0       # Standard deviation of the noise

I have 50 features, x's I put into my model, but just 10 of them are different from zero (the other 40 are equal to zero)

# Generate feature matrix X
# Each column is a feature drawn from standard normal distribution
X = np.random.randn(n_samples, n_features)

# Create the true coefficient vector (beta)
# Most coefficients are zero (sparse model)
true_coefficients = np.zeros(n_features)

# Randomly select which features will have non-zero coefficients
informative_features = np.random.choice(n_features, n_informative, replace=False)
print(f"True informative features indices: {sorted(informative_features)}")

# Assign non-zero values to selected coefficients
# Values are drawn from a normal distribution with larger variance
for idx in informative_features:
    true_coefficients[idx] = np.random.randn() * 3

# Generate the response variable Y
# Y = X * beta + noise
Y = X @ true_coefficients + np.random.randn(n_samples) * noise_level

# Save the data and true coefficients for later analysis
data_dict = {
    'X': X,
    'Y': Y,
    'true_coefficients': true_coefficients,
    'informative_features': informative_features
}

# Create a DataFrame to better visualize the coefficients
coef_df = pd.DataFrame({
    'feature_index': range(n_features),
    'true_coefficient': true_coefficients
})

# Show the non-zero coefficients
print("\nNon-zero coefficients:")
print(coef_df[coef_df['true_coefficient'] != 0])

# Split the data into training and testing sets
# We use 70% for training and 30% for testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Standardize the features (important for regularized regression)
# Fit the scaler on training data and apply to both train and test
scaler = StandardScaler()   
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

The function StandardScaler() means subtract the mean and divide by standard deviation, X_train_scaled and X_test_scaled are the standardized versions of the x's.
I have 140 observations in the training set and 60 observations in the test set. 

# Define different alpha values to test
# Alpha controls the strength of regularization
alphas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]

# Estimate Lasso for each alpha
lasso_results = {}

for alpha in alphas:
    # Create and fit Lasso model
    # max_iter: maximum number of iterations for optimization
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, Y_train)
    
    # Make predictions
    Y_train_pred = lasso.predict(X_train_scaled)
    Y_test_pred = lasso.predict(X_test_scaled)
    
    # Calculate metrics
    train_mse = mean_squared_error(Y_train, Y_train_pred)
    test_mse = mean_squared_error(Y_test, Y_test_pred)
    train_r2 = r2_score(Y_train, Y_train_pred)
    test_r2 = r2_score(Y_test, Y_test_pred)
    
    # Count non-zero coefficients
    n_nonzero = np.sum(lasso.coef_ != 0)
    
    # Store results
    lasso_results[alpha] = {
        'model': lasso,
        'coefficients': lasso.coef_,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'n_nonzero_coef': n_nonzero
    }
    
    print(f"\nAlpha = {alpha}")
    print(f"  Non-zero coefficients: {n_nonzero}")
    print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

    Alpha= 0.0001 (almost no penalization), the train MSE= 0.6, test MSE = 1.4, I have 140 observations, 50 x’s that I’m using (there are 3 observations per parameter to estimate). In sample, my model does well, out of sample doesn’t do so well, in sample the error is smaller than out of sample. I increase alpha, the train MSE is increasing (in sample test is increasing), and the test MSE is  
    decreasing, I’m reducing overfitting, I’m increasing regularization. When alpha is equal to 0.5, I get a very big train MSE, but also a big test MSE.  If there is an optimal alpha, 
    the optimal alpha (true alpha) should be between 0.1 and 0.5, because otherwise I’m penalizing too much or I’m penalizing too little. 

    # Store Ridge results for comparison
ridge_results = {}

for alpha in alphas:
    # Create and fit Ridge model
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, Y_train)
    
    # Make predictions
    Y_train_pred = ridge.predict(X_train_scaled)
    Y_test_pred = ridge.predict(X_test_scaled)
    
    # Calculate metrics
    train_mse = mean_squared_error(Y_train, Y_train_pred)
    test_mse = mean_squared_error(Y_test, Y_test_pred)
    train_r2 = r2_score(Y_train, Y_train_pred)
    test_r2 = r2_score(Y_test, Y_test_pred)
    
    # For Ridge, count "effectively zero" coefficients (very small)
    threshold = 0.001
    n_small = np.sum(np.abs(ridge.coef_) < threshold)
    
    # Store results
    ridge_results[alpha] = {
        'model': ridge,
        'coefficients': ridge.coef_,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'n_small_coef': n_small
    }
    
    print(f"\nAlpha = {alpha}")
    print(f"  Coefficients < {threshold}: {n_small}")
    print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")


    With Ridge regression, always 50 coefficients are different from zero

    # Select a specific alpha for detailed comparison
selected_alpha = 0.1

# Get the coefficients for the selected alpha
lasso_coef = lasso_results[selected_alpha]['coefficients']
ridge_coef = ridge_results[selected_alpha]['coefficients']

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Lasso coefficients vs True coefficients
ax1 = axes[0, 0]
ax1.scatter(true_coefficients, lasso_coef, alpha=0.6)
ax1.plot([-5, 5], [-5, 5], 'r--', label='Perfect recovery')
ax1.set_xlabel('True Coefficients')
ax1.set_ylabel('Lasso Coefficients')
ax1.set_title(f'Lasso Coefficient Recovery (α={selected_alpha})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Ridge coefficients vs True coefficients
ax2 = axes[0, 1]
ax2.scatter(true_coefficients, ridge_coef, alpha=0.6)
ax2.plot([-5, 5], [-5, 5], 'r--', label='Perfect recovery')
ax2.set_xlabel('True Coefficients')
ax2.set_ylabel('Ridge Coefficients')
ax2.set_title(f'Ridge Coefficient Recovery (α={selected_alpha})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Coefficient path for Lasso
ax3 = axes[1, 0]
for idx in informative_features:
    coef_path = [lasso_results[alpha]['coefficients'][idx] for alpha in alphas]
    ax3.plot(alphas, coef_path, 'b-', linewidth=2, alpha=0.8)
# Plot non-informative features in lighter color
for idx in range(n_features):
    if idx not in informative_features:
        coef_path = [lasso_results[alpha]['coefficients'][idx] for alpha in alphas]
        ax3.plot(alphas, coef_path, 'gray', linewidth=0.5, alpha=0.3)
ax3.set_xscale('log')
ax3.set_xlabel('Alpha (log scale)')
ax3.set_ylabel('Coefficient Value')
ax3.set_title('Lasso Coefficient Path')
ax3.grid(True, alpha=0.3)

# Plot 4: Number of non-zero coefficients vs alpha
ax4 = axes[1, 1]
nonzero_counts = [lasso_results[alpha]['n_nonzero_coef'] for alpha in alphas]
ax4.plot(alphas, nonzero_counts, 'o-', linewidth=2, markersize=8)
ax4.axhline(y=n_informative, color='r', linestyle='--', 
            label=f'True number ({n_informative})')
ax4.set_xscale('log')
ax4.set_xlabel('Alpha (log scale)')
ax4.set_ylabel('Number of Non-zero Coefficients')
ax4.set_title('Sparsity vs Regularization Strength')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Use LassoCV for automatic alpha selection
from sklearn.linear_model import LassoCV

# Define a range of alphas to test
alphas_cv = np.linspace(0.0001, 0.3, 50)

# Perform cross-validation
# cv=5 means 5-fold cross-validation
lasso_cv = LassoCV(alphas=alphas_cv, cv=5, max_iter=10000)
lasso_cv.fit(X_train_scaled, Y_train)

# Get the optimal alpha
optimal_alpha = lasso_cv.alpha_
print(f"Optimal alpha from cross-validation: {optimal_alpha:.4f}")

# Evaluate the model with optimal alpha
Y_test_pred_cv = lasso_cv.predict(X_test_scaled)
test_mse_cv = mean_squared_error(Y_test, Y_test_pred_cv)
test_r2_cv = r2_score(Y_test, Y_test_pred_cv)

print(f"Test MSE with optimal alpha: {test_mse_cv:.4f}")
print(f"Test R² with optimal alpha: {test_r2_cv:.4f}")


# Plot the cross-validation curve
plt.figure(figsize=(10, 6))
plt.errorbar(lasso_cv.alphas_, lasso_cv.mse_path_.mean(axis=1), 
            yerr=lasso_cv.mse_path_.std(axis=1), 
            label='Mean CV MSE ± 1 std')
plt.axvline(x=optimal_alpha, color='r', linestyle='--', 
           label=f'Optimal α = {optimal_alpha:.4f}')
plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Mean Squared Error')
plt.title('Cross-Validation Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

For each alpha I break the data randomly into 5 backets, then I use 4 backets to estimate, 1 backet to evaluate the model, and I do this for all the possible combinations of the backets, 
then I take the average of the MSE, I take the alpha that gives a lower MSE out of sample. When you do fit, instead of fitting only one Lasso, he fits all the 50 Lasso, one for each alpha 
and does the cross-validation. The optimal alpha is the alpha that gives the smallest cross-validated MSE.For each alpha, the blue line is the cross-validated MSE. The mean squared error 
decreases up to a point, and then starts to increase. The optimal alpha is 0.07 with a MSE of 1.044 on the test set, with the R^2= 0.99. 

# Create a summary comparison
summary_data = []

# Add Lasso results
for alpha in alphas:
    summary_data.append({
        'Method': 'Lasso',
        'Alpha': alpha,
        'Test MSE': lasso_results[alpha]['test_mse'],
        'Test R²': lasso_results[alpha]['test_r2'],
        'Non-zero Coefficients': lasso_results[alpha]['n_nonzero_coef']
    })

# Add Ridge results
for alpha in alphas:
    summary_data.append({
        'Method': 'Ridge',
        'Alpha': alpha,
        'Test MSE': ridge_results[alpha]['test_mse'],
        'Test R²': ridge_results[alpha]['test_r2'],
        'Non-zero Coefficients': n_features  # Ridge doesn't set coefficients to zero
    })

# Add CV Lasso result
summary_data.append({
    'Method': 'Lasso (CV)',
    'Alpha': optimal_alpha,
    'Test MSE': test_mse_cv,
    'Test R²': test_r2_cv,
    'Non-zero Coefficients': np.sum(lasso_cv.coef_ != 0)
})

summary_df = pd.DataFrame(summary_data)
print("\nModel Comparison Summary:")
print(summary_df)

EXERCISE 1

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set seed and parameters
np.random.seed(42)
n_samples = 100
n_features = 50
n_informative = 10
noise_level = 1.0

# Generate data
X = np.random.randn(n_samples, n_features)
true_coefficients = np.zeros(n_features)
informative_features = np.random.choice(n_features, n_informative, replace=False)
for i in informative_features:
    true_coefficients[i] = np.random.randn() * 3
Y = X @ true_coefficients + np.random.randn(n_samples) * noise_level

# Split and scale
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lasso with CV
alphas = np.logspace(-4, 0.3, 50)
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
lasso_cv.fit(X_train_scaled, Y_train)

# Evaluate feature recovery
estimated_support = set(np.where(lasso_cv.coef_ != 0)[0])
true_support = set(informative_features)
true_positives = len(estimated_support & true_support)
false_positives = len(estimated_support - true_support)

# Print summary
print(f"Sample size: {n_samples}")
print(f"Optimal alpha: {lasso_cv.alpha_:.5f}")
print(f"True Positives (TP): {true_positives} / {n_informative}")
print(f"False Positives (FP): {false_positives}")
print(f"Test R²: {r2_score(Y_test, lasso_cv.predict(X_test_scaled)):.4f}")
print(f"Test MSE: {mean_squared_error(Y_test, lasso_cv.predict(X_test_scaled)):.4f}")

# Plot recovery
plt.figure(figsize=(10, 6))
plt.scatter(true_coefficients, lasso_cv.coef_, alpha=0.7)
plt.axline((0, 0), slope=1, color='r', linestyle='--', label='Perfect recovery')
plt.xlabel('True Coefficients')
plt.ylabel('Estimated Coefficients')
plt.title(f'Lasso Coefficient Recovery (n={n_samples})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set seed and parameters
np.random.seed(42)
n_samples = 200
n_features = 50
n_informative = 10
noise_level = 1.0

# Generate data
X = np.random.randn(n_samples, n_features)
true_coefficients = np.zeros(n_features)
informative_features = np.random.choice(n_features, n_informative, replace=False)
for i in informative_features:
    true_coefficients[i] = np.random.randn() * 3
Y = X @ true_coefficients + np.random.randn(n_samples) * noise_level

# Split and scale
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lasso with CV
alphas = np.logspace(-4, 0.3, 50)
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
lasso_cv.fit(X_train_scaled, Y_train)

# Evaluate feature recovery
estimated_support = set(np.where(lasso_cv.coef_ != 0)[0])
true_support = set(informative_features)
true_positives = len(estimated_support & true_support)
false_positives = len(estimated_support - true_support)

# Print summary
print(f"Sample size: {n_samples}")
print(f"Optimal alpha: {lasso_cv.alpha_:.5f}")
print(f"True Positives (TP): {true_positives} / {n_informative}")
print(f"False Positives (FP): {false_positives}")
print(f"Test R²: {r2_score(Y_test, lasso_cv.predict(X_test_scaled)):.4f}")
print(f"Test MSE: {mean_squared_error(Y_test, lasso_cv.predict(X_test_scaled)):.4f}")

# Plot recovery
plt.figure(figsize=(10, 6))
plt.scatter(true_coefficients, lasso_cv.coef_, alpha=0.7)
plt.axline((0, 0), slope=1, color='r', linestyle='--', label='Perfect recovery')
plt.xlabel('True Coefficients')
plt.ylabel('Estimated Coefficients')
plt.title(f'Lasso Coefficient Recovery (n={n_samples})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set seed and parameters
np.random.seed(42)
n_samples = 1000
n_features = 50
n_informative = 10
noise_level = 1.0

# Generate data
X = np.random.randn(n_samples, n_features)
true_coefficients = np.zeros(n_features)
informative_features = np.random.choice(n_features, n_informative, replace=False)
for i in informative_features:
    true_coefficients[i] = np.random.randn() * 3
Y = X @ true_coefficients + np.random.randn(n_samples) * noise_level

# Split and scale
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lasso with CV
alphas = np.logspace(-4, 0.3, 50)
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
lasso_cv.fit(X_train_scaled, Y_train)

# Evaluate feature recovery
estimated_support = set(np.where(lasso_cv.coef_ != 0)[0])
true_support = set(informative_features)
true_positives = len(estimated_support & true_support)
false_positives = len(estimated_support - true_support)

# Print summary
print(f"Sample size: {n_samples}")
print(f"Optimal alpha: {lasso_cv.alpha_:.5f}")
print(f"True Positives (TP): {true_positives} / {n_informative}")
print(f"False Positives (FP): {false_positives}")
print(f"Test R²: {r2_score(Y_test, lasso_cv.predict(X_test_scaled)):.4f}")
print(f"Test MSE: {mean_squared_error(Y_test, lasso_cv.predict(X_test_scaled)):.4f}")

# Plot recovery
plt.figure(figsize=(10, 6))
plt.scatter(true_coefficients, lasso_cv.coef_, alpha=0.7)
plt.axline((0, 0), slope=1, color='r', linestyle='--', label='Perfect recovery')
plt.xlabel('True Coefficients')
plt.ylabel('Estimated Coefficients')
plt.title(f'Lasso Coefficient Recovery (n={n_samples})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

EXERCISE 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Parameters
n_samples = 200        # Number of observations
n_features = 50        # Total number of features
n_informative = 5      # Changed here to 5
noise_level = 1.0      # Noise std dev

# Generate data
X = np.random.randn(n_samples, n_features)
true_coefficients = np.zeros(n_features)
informative_features = np.random.choice(n_features, n_informative, replace=False)
print(f"True informative features indices: {sorted(informative_features)}")

for idx in informative_features:
    true_coefficients[idx] = np.random.randn() * 3

Y = X @ true_coefficients + np.random.randn(n_samples) * noise_level

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test alphas
alphas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
lasso_results = {}

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, Y_train)
    
    Y_test_pred = lasso.predict(X_test_scaled)
    test_mse = mean_squared_error(Y_test, Y_test_pred)
    test_r2 = r2_score(Y_test, Y_test_pred)
    n_nonzero = np.sum(lasso.coef_ != 0)
    
    lasso_results[alpha] = {
        'model': lasso,
        'coefficients': lasso.coef_,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'n_nonzero_coef': n_nonzero
    }
    
    print(f"\nAlpha = {alpha}")
    print(f"  Non-zero coefficients: {n_nonzero}")
    print(f"  Test MSE: {test_mse:.4f}")
    print(f"  Test R²: {test_r2:.4f}")

# Select alpha=0.1 for visualization
selected_alpha = 0.1
lasso_coef = lasso_results[selected_alpha]['coefficients']

# Plot true vs estimated coefficients
plt.figure(figsize=(8,6))
plt.scatter(true_coefficients, lasso_coef, alpha=0.7)
plt.plot([-5,5], [-5,5], 'r--', label='Perfect recovery')
plt.xlabel('True Coefficients')
plt.ylabel('Lasso Coefficients')
plt.title(f'Lasso Coefficient Recovery (n_informative={n_informative}, α={selected_alpha})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Parameters
n_samples = 200        # Number of observations
n_features = 50        # Total number of features
n_informative = 20     # Changed here to 20
noise_level = 1.0      # Noise std dev

# Generate data
X = np.random.randn(n_samples, n_features)
true_coefficients = np.zeros(n_features)
informative_features = np.random.choice(n_features, n_informative, replace=False)
print(f"True informative features indices: {sorted(informative_features)}")

for idx in informative_features:
    true_coefficients[idx] = np.random.randn() * 3

Y = X @ true_coefficients + np.random.randn(n_samples) * noise_level

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test alphas
alphas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
lasso_results = {}

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, Y_train)
    
    Y_test_pred = lasso.predict(X_test_scaled)
    test_mse = mean_squared_error(Y_test, Y_test_pred)
    test_r2 = r2_score(Y_test, Y_test_pred)
    n_nonzero = np.sum(lasso.coef_ != 0)
    
    lasso_results[alpha] = {
        'model': lasso,
        'coefficients': lasso.coef_,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'n_nonzero_coef': n_nonzero
    }
    
    print(f"\nAlpha = {alpha}")
    print(f"  Non-zero coefficients: {n_nonzero}")
    print(f"  Test MSE: {test_mse:.4f}")
    print(f"  Test R²: {test_r2:.4f}")

# Select alpha=0.1 for visualization
selected_alpha = 0.1
lasso_coef = lasso_results[selected_alpha]['coefficients']

# Plot true vs estimated coefficients
plt.figure(figsize=(8,6))
plt.scatter(true_coefficients, lasso_coef, alpha=0.7)
plt.plot([-5,5], [-5,5], 'r--', label='Perfect recovery')
plt.xlabel('True Coefficients')
plt.ylabel('Lasso Coefficients')
plt.title(f'Lasso Coefficient Recovery (n_informative={n_informative}, α={selected_alpha})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Parameters
n_samples = 200
n_features = 50
n_informative = 50  # All features informative now
noise_level = 1.0

# Generate data
X = np.random.randn(n_samples, n_features)
true_coefficients = np.zeros(n_features)
informative_features = np.arange(n_features)  # all features informative

for idx in informative_features:
    true_coefficients[idx] = np.random.randn() * 3

Y = X @ true_coefficients + np.random.randn(n_samples) * noise_level

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Alphas to test
alphas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
lasso_results = {}

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, Y_train)
    
    Y_test_pred = lasso.predict(X_test_scaled)
    test_mse = mean_squared_error(Y_test, Y_test_pred)
    test_r2 = r2_score(Y_test, Y_test_pred)
    n_nonzero = np.sum(lasso.coef_ != 0)
    
    lasso_results[alpha] = {
        'model': lasso,
        'coefficients': lasso.coef_,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'n_nonzero_coef': n_nonzero
    }
    
    print(f"\nAlpha = {alpha}")
    print(f"  Non-zero coefficients: {n_nonzero}")
    print(f"  Test MSE: {test_mse:.4f}")
    print(f"  Test R²: {test_r2:.4f}")

# Plot coefficient recovery for alpha = 0.1
selected_alpha = 0.1
lasso_coef = lasso_results[selected_alpha]['coefficients']

plt.figure(figsize=(8,6))
plt.scatter(true_coefficients, lasso_coef, alpha=0.7)
plt.plot([-10,10], [-10,10], 'r--', label='Perfect recovery')
plt.xlabel('True Coefficients')
plt.ylabel('Lasso Coefficients')
plt.title(f'Lasso Coefficient Recovery (n_informative={n_informative}, α={selected_alpha})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

 Ridge starts working better than Lasso, when we have many informative features, (many features that have β different from zero), while Lasso starts working better when we have lots of features that are not important, are not informative (when β’s are zero or very close to zero). 

EXERCISE 3

import numpy as np

# Set parameters
n_samples = 200
n_features = 50

# Start with identity matrix (no correlation)
correlation_matrix = np.eye(n_features)

# Add correlations between nearby features
# Here: Add correlation of 0.6 between neighboring features
for i in range(n_features - 1):
    correlation_matrix[i, i+1] = 0.6
    correlation_matrix[i+1, i] = 0.6

# Optionally check if the correlation matrix is positive semi-definite
# If not, np.random.multivariate_normal will throw an error
eigvals = np.linalg.eigvals(correlation_matrix)
if np.all(eigvals > 0):
    print("Correlation matrix is positive definite.")

# Generate multivariate normal data (correlated features)
X = np.random.multivariate_normal(
    mean=np.zeros(n_features), 
    cov=correlation_matrix, 
    size=n_samples
)

print("Shape of correlated X:", X.shape)




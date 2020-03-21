import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import linear_bandit
import linear_regression
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.linear_model import LinearRegression, LogisticRegression


def load_data_frame():
    path = "additional/data/warfarin.csv"
    df = pd.read_csv(path)
    df = df[df['Therapeutic Dose of Warfarin'].notna()]
    df = df.sample(frac=1)
    return df


def get_doses(df):
    target = df['Therapeutic Dose of Warfarin'].to_numpy()
    return target


def get_rewards(df):
    entries = df.shape[0]
    target = df['Therapeutic Dose of Warfarin'].to_numpy()
    rewards = np.zeros((entries, 3))
    for i in range(entries):
        if target[i] < 21.0:
            rewards[i, 1] = -1.0
            rewards[i, 2] = -1.0
        elif 21.0 <= target[i] <= 49.0:
            rewards[i, 0] = -1.0
            rewards[i, 2] = -1.0
        else:
            rewards[i, 0] = -1.0
            rewards[i, 1] = -1.0
    return rewards

def get_reward_class(df):
    entries = df.shape[0]
    target = df['Therapeutic Dose of Warfarin'].to_numpy()
    rewards = np.zeros((entries, 3))
    for i in range(entries):
        if target[i] < 21.0:
            rewards[i, 0] = 1
            rewards[i, 1] = 0
            rewards[i, 2] = 0
        elif 21.0 <= target[i] <= 49.0:
            rewards[i, 0] = 0
            rewards[i, 1] = 1
            rewards[i, 2] = 0
        else:
            rewards[i, 0] = 0
            rewards[i, 1] = 0
            rewards[i, 2] = 1
    return rewards

def get_reward_class_sparse(df):
    entries = df.shape[0]
    target = df['Therapeutic Dose of Warfarin'].to_numpy()
    rewards = np.zeros((entries))
    for i in range(entries):
        if target[i] < 21.0:
            rewards[i] = 0
        elif 21.0 <= target[i] <= 49.0:
            rewards[i] = 1
        else:
            rewards[i] = 2
    return rewards

def dose_to_action(target):
    if target < 21.0:
        return 0
    elif 21.0 <= target <= 49.0:
        return 1
    else:
        return 2

def dose_to_action_idx(dose):
    idx = 2
    if dose < 21.0:
        idx = 0
    elif 21.0 <= dose <= 49.0:
        idx = 1
    return idx


def get_numpy_data(df):
    entries = df.shape[0]

    medsdf = df.loc[:, ['Aspirin', 'Acetaminophen or Paracetamol (Tylenol)',
                       'Was Dose of Acetaminophen or Paracetamol (Tylenol) >1300mg/day',
                       'Simvastatin (Zocor)', 'Atorvastatin (Lipitor)', 'Fluvastatin (Lescol)',
                       'Lovastatin (Mevacor)', 'Pravastatin (Pravachol)',
                       'Rosuvastatin (Crestor)', 'Cerivastatin (Baycol)',
                       'Amiodarone (Cordarone)', 'Carbamazepine (Tegretol)',
                       'Phenytoin (Dilantin)', 'Rifampin or Rifampicin',
                       'Sulfonamide Antibiotics', 'Macrolide Antibiotics',
                       'Anti-fungal Azoles', 'Herbal Medications, Vitamins, Supplements', ]]
    genotypes = df.loc[:, ['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T',
       'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C',
       'VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G',
       'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G',
       'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G',
       'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G',
       'VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C' ]]

    gender_raw = df['Gender'].fillna(value='male').to_numpy()
    gender = LabelEncoder().fit_transform(gender_raw)
    race = pd.get_dummies(df['Race'], prefix=['Race']).to_numpy()
    ethnicity = pd.get_dummies(df['Ethnicity'], prefix=['Ethnicity']).to_numpy()
    le = LabelEncoder()
    le.fit(['0 - 9', '10 - 19', '20 - 29', '30 - 39', '40 - 49', '50 - 59', '60 - 69', '70 - 79', '80 - 89', '90+'])

    # Hack solution: set NA to age 50
    # better to do an averaging
    age_raw = df['Age'].fillna(value='50 - 59').to_numpy()
    age = le.transform(age_raw)
    height = df['Height (cm)'].fillna(value=170).to_numpy()
    weight = df['Weight (kg)'].fillna(value=90).to_numpy()
    indication = np.zeros((entries, 8))
    for i in range(entries):
        ind = df['Indication for Warfarin Treatment'].iloc[[i]].item()
        if pd.notna(ind):
            for j in range(8):
                indication[i, j] = 1 if str(j+1) in ind else 0
        else:
            indication[i, 7] = 1
    diabetes = df['Diabetes'].fillna(value=0.0).to_numpy()
    heart_failure = df['Congestive Heart Failure and/or Cardiomyopathy'].fillna(value=0.0).to_numpy()
    valve = df['Valve Replacement'].fillna(value=0.0).to_numpy()
    meds = medsdf.fillna(value=0.0).to_numpy()
    smoker = df['Current Smoker'].fillna(value=0.0).to_numpy()
    cyp2C9 = pd.get_dummies(df['Cyp2C9 genotypes'], prefix=['Cyp2C9 genotypes']).to_numpy()
    vk = []
    for gen in genotypes.columns:
        vk.append(pd.get_dummies(df[gen], prefix=[gen]).to_numpy())

    data = np.column_stack((gender, race, ethnicity, age, height, weight, indication, diabetes, heart_failure, valve,
                            meds, smoker, cyp2C9))
    for i in range(len(genotypes.columns)):
        data = np.column_stack((data, vk[i]))

    return data


def test_basic(df):
    rewards = get_rewards(df)
    trials = rewards.shape[0]


    X = get_numpy_data(df)
    Y = get_rewards(df)
    oracle_preds = linear_regression.get_oracle_preds(X, Y)
    oracle_max = np.max(oracle_preds, axis=1)

    regret = [0]
    incorrect_ticks = []

    total_attempts = 0
    correct_actions = 0
    incorrect_fraction = 0
    for i in range(trials):
        if rewards[i, 1] == 0:
            correct_actions += 1
        total_attempts += 1
        incorrect_fraction = 1 - correct_actions * 1. / total_attempts

        regret.append(regret[i] + np.mean(oracle_max - oracle_preds[:, 1]))
        incorrect_ticks.append(incorrect_fraction)

    return 1-incorrect_fraction, regret, incorrect_ticks


def test_clinical(df):
    rewards = get_rewards(df)
    trials = rewards.shape[0]
    enzymes = df.loc[:, ['Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 'Rifampin or Rifampicin']].fillna(value=0.0)

    X = get_numpy_data(df)
    Y = get_rewards(df)
    oracle_preds = linear_regression.get_oracle_preds(X, Y)
    oracle_max = np.max(oracle_preds, axis=1)

    incorrect_ticks = []
    total_attempts = 0
    correct_actions = 0
    incorrect_fraction = 0
    clinical_actions = []
    for i in range(trials):
        entry = df.iloc[[i]]
        enzyme_entry = enzymes.iloc[[i]]
        age = int(entry['Age'].fillna(value='5').item()[0])
        height = int(entry['Height (cm)'].fillna(value=170).item())
        weight = int(entry['Weight (kg)'].fillna(value=85).item())
        race = entry['Race'].fillna(value='Unknown').item()
        enzyme_status = 1.0 if 1.0 in enzyme_entry.values else 0.0
        amiodarone = entry['Amiodarone (Cordarone)'].fillna(value=0.0).item()
        dose = 4.0376 - 0.2546*age + 0.0118*height + 0.0134*weight + 1.2799*enzyme_status - 0.5695*amiodarone
        if race == 'Black or African American':
            dose += 0.4060
        elif race == 'Asian':
            dose -= 0.6752
        elif race == 'Unknown':
            dose += 0.0433
        dose = dose**2

        action = dose_to_action_idx(dose)
        clinical_actions.append(action)
        if rewards[i, action] == 0:
            correct_actions += 1
        total_attempts += 1

        incorrect_fraction = 1 - correct_actions * 1. / total_attempts
        incorrect_ticks.append(incorrect_fraction)

    # Expected Regret Per Trial
    expected_sum = 0
    for j, idx in enumerate(clinical_actions):
        expected_sum += oracle_max[j] - oracle_preds[j][idx]
    expected_sum /= trials

    # Regret Ticks
    regret = [0]
    for i in range(trials):
        regret.append(regret[i] + expected_sum)

    return 1 - incorrect_fraction, regret, incorrect_ticks


def test_linear_bandit(df, lr=0.5):
    X = get_numpy_data(df)
    Y = get_rewards(df)
    oracle_preds = linear_regression.get_oracle_preds(X, Y)
    oracle_max = np.max(oracle_preds, axis=1)

    bandit = linear_bandit.LinearBandit(feature_dims=X.shape[1], learning_rate=lr)
    regret = [0]
    incorrect_ticks = []

    total_attempts = 0
    correct_actions = 0
    for t in range(X.shape[0]):
        # update bandit
        action = bandit.take_action(X[t].reshape(-1, 1))
        bandit.update_arm(action, X[t].reshape(-1, 1), Y[t][action])

        # Performance
        if Y[t, action] == 0:
            correct_actions += 1
        total_attempts += 1

        incorrect_fraction = 1 - correct_actions * 1. / total_attempts
        incorrect_ticks.append(incorrect_fraction)

        # Regret
        bandit_preds = bandit.evaluate_beta(X)
        expected_sum = 0
        for j, idx in enumerate(bandit_preds):
            expected_sum += oracle_max[j] - oracle_preds[j][idx]
        expected_sum /= X.shape[0]
        regret.append(regret[t] + expected_sum)

    return 1 - incorrect_ticks[-1], regret, incorrect_ticks


def test_supervised(df):
    X = get_numpy_data(df)
    Y = get_rewards(df)
    doses = get_doses(df)

    oracle_preds = linear_regression.get_oracle_preds(X, Y)
    oracle_max = np.max(oracle_preds, axis=1)

    regret = [0]
    incorrect_ticks = []


    total_attempts = 0
    correct_actions = 0
    logit = LogisticRegression()
    # sparse_rewards = get_reward_class(df)
    sparse_rewards = get_reward_class_sparse(df)

    # reg = LinearRegression()
    for t in range(1, X.shape[0]):
        reg = LinearRegression().fit(X[:t], Y[:t])
        # dose based
        # reg.fit(X[:t], dose[:t])

        action = np.argmax(np.squeeze(reg.predict(X[t].reshape(1, -1))))

        # action = np.squeeze(reg.predict(X[t].reshape(1, -1)))
        # action = dose_to_action(dose)

        # Performance
        if Y[t, action] == 0:
            correct_actions += 1
        total_attempts += 1

        incorrect_fraction = 1 - correct_actions * 1. / total_attempts
        incorrect_ticks.append(incorrect_fraction)

        # Regret
        preds = np.argmax(reg.predict(X), axis=1)

        # dose based
        # preds = reg.predict(X)

        expected_sum = 0
        for j, idx in enumerate(preds):
            # idx = dose_to_action(idx)
            expected_sum += oracle_max[j] - oracle_preds[j][idx]

        expected_sum /= X.shape[0]
        regret.append(regret[t-1] + expected_sum)

        # if t % 100 == 0:
        #     print("Timestep:", t, "Regret:", regret[t-1], "Error", incorrect_ticks[t-1])

    return 1 - incorrect_ticks[-1], regret, incorrect_ticks


def mean_conf_interval(data, confidence=0.95):
    n = data.shape[0]
    m, se = np.mean(data), st.sem(data)
    h = se * st.t.ppf((1+confidence) / 2., n-1)
    return m, m-h, m+h



def run_trials():
    basic_reg = []
    basic_err = []

    clinical_reg = []
    clinical_err = []

    bandit_reg = []
    bandit_err = []

    for i in range(20):
        df = load_data_frame()
        print("Epoch", i + 1)
        basic_score, basic_regret, basic_perf = test_basic(df)
        basic_reg.append(basic_regret)
        basic_err.append(basic_perf)
        print("Results of Fixed-dose:", basic_score)

        clinical_score, clinical_regret, clin_perf = test_clinical(df)
        clinical_reg.append(clinical_regret)
        clinical_err.append(clin_perf)
        print("Results of Clinical-dose:", clinical_score)

        bandit_score, bandit_regret, bandit_perf = test_linear_bandit(df)
        bandit_reg.append(bandit_regret)
        bandit_err.append(bandit_perf)
        print("Results of Bandit-dose:", bandit_score)

    basic_reg = np.array(basic_reg)
    basic_reg = np.delete(basic_reg, 0, 1)
    basic_err = np.array(basic_err)
    ticks = basic_reg.shape[1]

    clinical_reg = np.array(clinical_reg)
    clinical_reg = np.delete(clinical_reg, 0, 1)
    clinical_err = np.array(clinical_err)

    bandit_reg = np.array(bandit_reg)
    bandit_reg = np.delete(bandit_reg, 0, 1)
    bandit_err = np.array(bandit_err)

    basic_err_mean = []
    basic_err_plus = []
    basic_err_minus = []
    basic_reg_mean = []
    basic_reg_plus = []
    basic_reg_minus = []

    clinical_err_mean = []
    clinical_err_plus = []
    clinical_err_minus = []
    clinical_reg_mean = []
    clinical_reg_plus = []
    clinical_reg_minus = []

    bandit_err_mean = []
    bandit_err_plus = []
    bandit_err_minus = []
    bandit_reg_mean = []
    bandit_reg_plus = []
    bandit_reg_minus = []
    for i in range(ticks):
        mean, plus, minus = mean_conf_interval(basic_err[:, i])
        basic_err_mean.append(mean)
        basic_err_plus.append(plus)
        basic_err_minus.append(minus)
        mean, plus, minus = mean_conf_interval(basic_reg[:, i])
        basic_reg_mean.append(mean)
        basic_reg_plus.append(plus)
        basic_reg_minus.append(minus)

        mean, plus, minus = mean_conf_interval(clinical_err[:, i])
        clinical_err_mean.append(mean)
        clinical_err_plus.append(plus)
        clinical_err_minus.append(minus)
        mean, plus, minus = mean_conf_interval(clinical_reg[:, i])
        clinical_reg_mean.append(mean)
        clinical_reg_plus.append(plus)
        clinical_reg_minus.append(minus)

        mean, plus, minus = mean_conf_interval(bandit_err[:, i])
        bandit_err_mean.append(mean)
        bandit_err_plus.append(plus)
        bandit_err_minus.append(minus)
        mean, plus, minus = mean_conf_interval(bandit_reg[:, i])
        bandit_reg_mean.append(mean)
        bandit_reg_plus.append(plus)
        bandit_reg_minus.append(minus)

    # Reg Graph
    plt.plot(range(ticks), basic_reg_mean, color='blue', label="Basic Regret")
    plt.fill_between(range(ticks), basic_reg_plus, basic_reg_minus, color='blue', alpha=0.5)
    plt.plot(range(ticks), clinical_reg_mean, color='green', label="Clinical Regret")
    plt.fill_between(range(ticks), clinical_reg_plus, clinical_reg_minus, color='green', alpha=0.5)
    plt.plot(range(ticks), bandit_reg_mean, color='red', label="Bandit Regret")
    plt.fill_between(range(ticks), bandit_reg_plus, bandit_reg_minus, color='red', alpha=0.5)
    plt.title("Regret vs Time")
    plt.legend()
    plt.savefig("regret.png")
    plt.show()
    plt.close()

    # Error Graph
    plt.plot(range(ticks), basic_err_mean, color='blue', label="Basic Error")
    plt.fill_between(range(ticks), basic_err_plus, basic_err_minus, color='blue', alpha=0.5)
    plt.plot(range(ticks), clinical_err_mean, color='green', label="Clinical Error")
    plt.fill_between(range(ticks), clinical_err_plus, clinical_err_minus, color='green', alpha=0.5)
    plt.plot(range(ticks), bandit_err_mean, color='red', label="Bandit Error")
    plt.fill_between(range(ticks), bandit_err_plus, bandit_err_minus, color='red', alpha=0.5)
    plt.title("Incorrect Dose vs Time")
    plt.legend()
    plt.savefig("error.png")
    plt.show()
    plt.close()


def run_supervised():
    bandit_reg = []
    bandit_err = []

    sup_reg = []
    sup_err = []
    for i in range(20):
        df = load_data_frame()
        print("Epoch", i + 1)

        bandit_score, bandit_regret, bandit_perf = test_linear_bandit(df)
        bandit_reg.append(bandit_regret)
        bandit_err.append(bandit_perf)
        print("Results of Bandit-dose:", bandit_score, bandit_regret[-1])

        sup_score, sup_regret, sup_perf = test_supervised(df)
        sup_reg.append(sup_regret)
        sup_err.append(sup_perf)
        print("Results of Sup-dose:", sup_score, sup_regret[-1])


    bandit_reg = np.array(bandit_reg)
    bandit_reg = np.delete(bandit_reg, 0, 1)
    bandit_err = np.array(bandit_err)

    sup_reg = np.array(sup_reg)
    sup_reg = np.delete(sup_reg, 0, 1)
    sup_err = np.array(sup_err)
    ticks = sup_reg.shape[1]

    sup_err_mean = []
    sup_err_plus = []
    sup_err_minus = []
    sup_reg_mean = []
    sup_reg_plus = []
    sup_reg_minus = []

    bandit_err_mean = []
    bandit_err_plus = []
    bandit_err_minus = []
    bandit_reg_mean = []
    bandit_reg_plus = []
    bandit_reg_minus = []
    for i in range(ticks):

        mean, plus, minus = mean_conf_interval(sup_err[:, i])
        sup_err_mean.append(mean)
        sup_err_plus.append(plus)
        sup_err_minus.append(minus)
        mean, plus, minus = mean_conf_interval(sup_reg[:, i])
        sup_reg_mean.append(mean)
        sup_reg_plus.append(plus)
        sup_reg_minus.append(minus)

        mean, plus, minus = mean_conf_interval(bandit_err[:, i])
        bandit_err_mean.append(mean)
        bandit_err_plus.append(plus)
        bandit_err_minus.append(minus)
        mean, plus, minus = mean_conf_interval(bandit_reg[:, i])
        bandit_reg_mean.append(mean)
        bandit_reg_plus.append(plus)
        bandit_reg_minus.append(minus)

    # Reg Graph
    plt.plot(range(ticks), sup_reg_mean, color='blue', label="Supervised Regret")
    plt.fill_between(range(ticks), sup_reg_plus, sup_reg_minus, color='blue', alpha=0.5)
    plt.plot(range(ticks), bandit_reg_mean, color='red', label="Bandit Regret")
    plt.fill_between(range(ticks), bandit_reg_plus, bandit_reg_minus, color='red', alpha=0.5)
    plt.title("Regret vs Time")
    plt.legend()
    plt.savefig("regret_sup.png")
    plt.show()
    plt.close()

    # Error Graph
    plt.plot(range(ticks), sup_err_mean, color='blue', label="Supervised Error")
    plt.fill_between(range(ticks), sup_err_plus, sup_err_minus, color='blue', alpha=0.5)
    plt.plot(range(ticks), bandit_err_mean, color='red', label="Bandit Error")
    plt.fill_between(range(ticks), bandit_err_plus, bandit_err_minus, color='red', alpha=0.5)
    plt.title("Incorrect Dose vs Time")
    plt.legend()
    plt.savefig("error_sup.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # run_trials()
    run_supervised()

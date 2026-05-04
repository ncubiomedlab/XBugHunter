import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.neural_network import MLPClassifier

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))

    # Youden Index (J)
    roc = pd.DataFrame({'tf': pd.Series(tpr - fpr, index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.loc[(roc.tf - 0).abs().argsort()[-1:]]


    return roc_t.iloc[0, 1]  # 0=S 1=R




input_path = 'area//'
output_pth = 'area//'

outName ='indep_testing'


bac_abx = {'bac':('E_faecium','S_aureus','K_pneumoniae','A_baumannii_mdr','E_aerogenes','E_cloacae'),
    'abx':('VA','OX','CIP','CIP','CRO','CRO')}

acc_DT = []
sen_DT = []
spe_DT = []
auc_DT = []
cut_off_DT = []

acc_RF = []
sen_RF = []
spe_RF = []
auc_RF = []
cut_off_RF = []

acc_LR = []
sen_LR = []
spe_LR = []
auc_LR = []
cut_off_LR = []

acc_SVM = []
sen_SVM = []
spe_SVM = []
auc_SVM = []
cut_off_SVM = []

acc_NN = []
sen_NN = []
spe_NN = []
auc_NN = []
cut_off_NN = []

for BAC in range(0, len(bac_abx['bac'])):
    print(bac_abx['bac'][BAC])
    print(bac_abx['abx'][BAC])
    anti = bac_abx['abx'][BAC]

    FileName = bac_abx['bac'][BAC]
    training = pd.read_csv(input_path + 'kde_train_' + FileName + '_01.csv')
    testing = pd.read_csv(input_path + 'kde_test_' + FileName + '_01.csv')
    #training = pd.read_csv(input_path + 'bin_train_' + FileName + '.csv')
    #testing = pd.read_csv(input_path + 'bin_test_' + FileName + '.csv')

    y_train = training.Label
    X_train = training.drop("Label", axis=1)

    y_test = testing.Label
    X_test = testing.drop("Label", axis=1)

    model_DT = DecisionTreeClassifier(random_state=0, criterion='entropy')
    model_RF = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    model_LR = LogisticRegression(random_state=0, solver='saga')
    model_SVM = SVC(probability=True, gamma='scale', random_state=0)
    model_NN = MLPClassifier(solver='adam', hidden_layer_sizes=(1024, 512), random_state=1, activation='tanh',
                             learning_rate='adaptive',
                             early_stopping=True)

    model_DT.fit(X_train, y_train)
    model_RF.fit(X_train, y_train)
    model_LR.fit(X_train, y_train)
    model_SVM.fit(X_train, y_train)
    model_NN.fit(X_train, y_train)

    y_pred_prob_DT = model_DT.predict_proba(X_test)[:, 1]
    y_pred_prob_RF = model_RF.predict_proba(X_test)[:, 1]
    y_pred_prob_LR = model_LR.predict_proba(X_test)[:, 1]
    y_pred_prob_SVM = model_SVM.predict_proba(X_test)[:, 1]
    y_pred_prob_NN = model_NN.predict_proba(X_test)[:, 1]

    opt_cut_DT = Find_Optimal_Cutoff(y_test, y_pred_prob_DT)
    opt_cut_RF = Find_Optimal_Cutoff(y_test, y_pred_prob_RF)
    opt_cut_LR = Find_Optimal_Cutoff(y_test, y_pred_prob_LR)
    opt_cut_SVM = Find_Optimal_Cutoff(y_test, y_pred_prob_SVM)
    opt_cut_NN = Find_Optimal_Cutoff(y_test, y_pred_prob_NN)

    cut_off_DT.append(opt_cut_DT)
    cut_off_RF.append(opt_cut_RF)
    cut_off_LR.append(opt_cut_LR)
    cut_off_SVM.append(opt_cut_SVM)
    cut_off_NN.append(opt_cut_NN)

    y_pred_DT = (y_pred_prob_DT >= opt_cut_DT).astype(int)
    y_pred_RF = (y_pred_prob_RF >= opt_cut_RF).astype(int)
    y_pred_LR = (y_pred_prob_LR >= opt_cut_LR).astype(int)
    y_pred_SVM = (y_pred_prob_SVM >= opt_cut_SVM).astype(int)
    y_pred_NN = (y_pred_prob_NN >= opt_cut_NN).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_DT).ravel()
    acc_DT.append((tp + tn) / (tp + tn + fp + fn))
    sen_DT.append(tp / (tp + fn))
    spe_DT.append(tn / (tn + fp))
    auc_DT.append(roc_auc_score(y_test, y_pred_prob_DT))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_RF).ravel()
    acc_RF.append((tp + tn) / (tp + tn + fp + fn))
    sen_RF.append(tp / (tp + fn))
    spe_RF.append(tn / (tn + fp))
    auc_RF.append(roc_auc_score(y_test, y_pred_prob_RF))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_LR).ravel()
    acc_LR.append((tp + tn) / (tp + tn + fp + fn))
    sen_LR.append(tp / (tp + fn))
    spe_LR.append(tn / (tn + fp))
    auc_LR.append(roc_auc_score(y_test, y_pred_prob_LR))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_SVM).ravel()
    acc_SVM.append((tp + tn) / (tp + tn + fp + fn))
    sen_SVM.append(tp / (tp + fn))
    spe_SVM.append(tn / (tn + fp))
    auc_SVM.append(roc_auc_score(y_test, y_pred_prob_SVM))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_NN).ravel()
    acc_NN.append((tp + tn) / (tp + tn + fp + fn))
    sen_NN.append(tp / (tp + fn))
    spe_NN.append(tn / (tn + fp))
    auc_NN.append(roc_auc_score(y_test, y_pred_prob_NN))

    predictions = pd.DataFrame()
    predictions['DT_pred_prob'] = y_pred_prob_DT
    predictions['DT_pred_bin'] = y_pred_DT

    predictions['RF_pred_prob'] = y_pred_prob_RF
    predictions['RF_pred_bin'] = y_pred_RF

    predictions['LR_pred_prob'] = y_pred_prob_LR
    predictions['LR_pred_bin'] = y_pred_LR

    predictions['SVM_pred_prob'] = y_pred_prob_SVM
    predictions['SVM_pred_bin'] = y_pred_SVM

    predictions['NN_pred_prob'] = y_pred_prob_NN
    predictions['NN_pred_bin'] = y_pred_NN

    predictions['True_Label'] = y_test
    predictions.to_csv(output_pth + FileName + '_indep_pred.csv', index=False)

    print("========================================")
    print("%.4f" % roc_auc_score(y_test, y_pred_prob_DT))
    print("%.4f" % roc_auc_score(y_test, y_pred_prob_RF))
    print("%.4f" % roc_auc_score(y_test, y_pred_prob_LR))
    print("%.4f" % roc_auc_score(y_test, y_pred_prob_SVM))
    print("%.4f" % roc_auc_score(y_test, y_pred_prob_NN))
    print("========================================")



final = pd.DataFrame()
final['DT_SEN'] = sen_DT
final['DT_SPE'] = spe_DT
final['DT_ACC'] = acc_DT
final['DT_AUC'] = auc_DT
final['DT_cut_off'] = cut_off_DT

final['RF_SEN'] = sen_RF
final['RF_SPE'] = spe_RF
final['RF_ACC'] = acc_RF
final['RF_AUC'] = auc_RF
final['RF_cut_off'] = cut_off_RF

final['LR_SEN'] = sen_LR
final['LR_SPE'] = spe_LR
final['LR_ACC'] = acc_LR
final['LR_AUC'] = auc_LR
final['LR_cut_off'] = cut_off_LR

final['SVM_SEN'] = sen_SVM
final['SVM_SPE'] = spe_SVM
final['SVM_ACC'] = acc_SVM
final['SVM_AUC'] = auc_SVM
final['SVM_cut_off'] = cut_off_SVM

final['NN_SEN'] = sen_NN
final['NN_SPE'] = spe_NN
final['NN_ACC'] = acc_NN
final['NN_AUC'] = auc_NN
final['NN_cut_off'] = cut_off_NN

final.to_csv(output_pth + outName +  '.csv', index=False)

print(final)




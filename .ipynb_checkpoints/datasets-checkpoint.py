from foldrm import Classifier
import numpy as np

def birds():
    attrs: ['wingspan', 'beak']
    nums: ['wingspan']
    model = Classifier(attrs=attrs, numeric=nums, label='predator')
    data = model.load_data('data/birds/birds.csv')
    print('\n% birds dataset', np.shape(data))
    return model, data



def acute():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    nums = ['a1']
    model = Classifier(attrs=attrs, numeric=nums, label='label')
    data = model.load_data('data/acute/acute.csv')
    print('\n% acute dataset', np.shape(data))
    return model, data


def adult():
    attrs = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship',
    'race','sex','capital_gain','capital_loss','hours_per_week','native_country']
    nums = ['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']
    model = Classifier(attrs=attrs, numeric=nums, label='label')
    data = model.load_data('data/adult/adult.csv')
    print('\n% adult dataset', np.shape(data))
    return model, data


def autism():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'age', 'gender', 'ethnicity', 'jaundice',
             'pdd', 'used_app_before', 'relation']
    nums = ['age']
    model = Classifier(attrs=attrs, numeric=nums, label='label')
    data = model.load_data('data/autism/autism.csv')
    print('\n% autism dataset', np.shape(data))
    return model, data


def breastw():
    attrs = ['clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 'marginal_adhesion',
    'single_epi_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='label')
    data = model.load_data('data/breastw/breastw.csv')
    print('\n% breastw dataset', np.shape(data))
    return model, data


def cars():
    attrs = ['buying', 'maint', 'doors', 'persons', 'lugboot', 'safety']
    model = Classifier(attrs=attrs, numeric=[], label='label')
    data = model.load_data('data/cars/cars.csv')
    print('\n% cars dataset', np.shape(data))
    return model, data


def credit():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15']
    nums = ['a2', 'a3', 'a8', 'a11', 'a14', 'a15']
    model = Classifier(attrs=attrs, numeric=nums, label='label')
    data = model.load_data('data/credit/credit.csv')
    print('\n% credit dataset', np.shape(data))
    return model, data


def heart():
    attrs = ['age', 'sex', 'chest_pain', 'blood_pressure', 'serum_cholestoral', 'fasting_blood_sugar',
    'resting_electrocardiographic_results', 'maximum_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak',
    'slope', 'major_vessels', 'thal']
    nums = ['age', 'blood_pressure', 'serum_cholestoral', 'maximum_heart_rate_achieved', 'oldpeak']
    model = Classifier(attrs=attrs, numeric=nums, label='label')
    data = model.load_data('data/heart/heart.csv')
    print('\n% heart dataset', np.shape(data))
    return model, data


def kidney():
    attrs = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv',
    'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    nums = ['age', 'bp', 'sg', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
    model = Classifier(attrs=attrs, numeric=nums, label='label')
    data = model.load_data('data/kidney/kidney.csv')
    print('\n% kidney dataset', np.shape(data))
    return model, data


def krkp():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16',
    'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a30', 'a31', 'a32',
    'a33', 'a34', 'a35', 'a36']
    model = Classifier(attrs=attrs, numeric=[], label='label')
    data = model.load_data('data/krkp/krkp.csv')
    print('\n% krkp dataset', np.shape(data))
    return model, data


def mushroom():
    attrs = ['cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment', 'gill_spacing',
    'gill_size', 'gill_color', 'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
    'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number', 'ring_type',
    'spore_print_color', 'population', 'habitat']
    model = Classifier(attrs=attrs, numeric=[], label='label')
    data = model.load_data('data/mushroom/mushroom.csv')
    print('\n% mushroom dataset', np.shape(data))
    return model, data


def sonar():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16',
    'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a30', 'a31', 'a32',
    'a33', 'a34', 'a35', 'a36', 'a37', 'a38', 'a39', 'a40', 'a41', 'a42', 'a43', 'a44', 'a45', 'a46', 'a47', 'a48',
    'a49', 'a50', 'a51', 'a52', 'a53', 'a54', 'a55', 'a56', 'a57', 'a58', 'a59', 'a60']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='label')
    data = model.load_data('data/sonar/sonar.csv')
    print('\n% sonar dataset', np.shape(data))
    return model, data


def voting():
    attrs = ['handicapped_infants', 'water_project_cost_sharing', 'budget_resolution', 'physician_fee_freeze',
    'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban', 'aid_to_nicaraguan_contras',
    'mx_missile', 'immigration', 'synfuels_corporation_cutback', 'education_spending', 'superfund_right_to_sue',
    'crime', 'duty_free_exports', 'export_administration_act_south_africa']
    model = Classifier(attrs=attrs, numeric=[], label='label')
    data = model.load_data('data/voting/voting.csv')
    print('\n% voting dataset', np.shape(data))
    return model, data


def ecoli():
    attrs = ['sn','mcg','gvh','lip','chg','aac','alm1','alm2']
    nums = ['mcg','gvh','lip','chg','aac','alm1','alm2']
    model = Classifier(attrs=attrs, numeric=nums, label='label')
    data = model.load_data('data/ecoli/ecoli.csv')
    print('\n% ecoli dataset', np.shape(data))
    return model, data


def ionosphere():
    attrs = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19',
    'c20','c21','c22','c23','c24','c25','c26','c27','c28','c29','c30','c31','c32','c33','c34']
    model = Classifier(attrs=attrs, numeric=attrs, label='label')
    data = model.load_data('data/ionosphere/ionosphere.csv')
    print('\n% ionosphere dataset', np.shape(data))
    return model, data


def wine():
    attrs = ['alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','tot_phenols','flavanoids',
    'nonflavanoid_phenols','proanthocyanins','color_intensity','hue','OD_of_diluted','proline']
    model = Classifier(attrs=attrs, numeric=attrs, label='label')
    data = model.load_data('data/wine/wine.csv')
    print('\n% wine dataset', np.shape(data))
    return model, data


def credit_card():
    attrs = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
    'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4',
    'PAY_AMT5','PAY_AMT6']
    nums = ['LIMIT_BAL','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1',
    'PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
    model = Classifier(attrs=attrs, numeric=nums, label='DEFAULT_PAYMENT')
    data = model.load_data('data/credit_card/credit_card.csv')
    print('\n% credit card dataset', np.shape(data))
    return model, data


def rain():
    attrs = ['Month','Day','Location','MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RainToday']
    nums = ['Month','Day','MinTemp','MaxTemp','Rainfall','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Temp9am','Temp3pm']
    model = Classifier(attrs=attrs, numeric=nums, label='RainTomorrow')
    data = model.load_data('data/rain/rain.csv')
    print('\n% rain dataset', np.shape(data))
    return model, data


def heloc():
    attrs = ['ExternalRiskEstimate','MSinceOldestTradeOpen','MSinceMostRecentTradeOpen','AverageMInFile','NumSatisfactoryTrades','NumTrades60Ever2DerogPubRec','NumTrades90Ever2DerogPubRec','PercentTradesNeverDelq','MSinceMostRecentDelq','MaxDelq2PublicRecLast12M','MaxDelqEver','NumTotalTrades','NumTradesOpeninLast12M','PercentInstallTrades','MSinceMostRecentInqexcl7days','NumInqLast6M','NumInqLast6Mexcl7days','NetFractionRevolvingBurden','NetFractionInstallBurden','NumRevolvingTradesWBalance','NumInstallTradesWBalance','NumBank2NatlTradesWHighUtilization','PercentTradesWBalance']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='RiskPerformance')
    data = model.load_data('data/heloc/heloc_dataset_v1.csv')
    print('\n% rain dataset', np.shape(data))
    return model, data


def avila():
    attrs = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10']
    nums = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10']
    model = Classifier(attrs=attrs, numeric=nums, label='class')
    data_train = model.load_data('data/avila/train.csv')
    data_test = model.load_data('data/avila/test.csv')
    print('\n% avila dataset train', np.shape(data_train), 'test', np.shape(data_test))
    return model, data_train, data_test


def titanic():
    attrs = ['Sex', 'Age', 'Number_of_Siblings_Spouses', 'Number_Of_Parents_Children', 'Fare', 'Class', 'Embarked']
    nums = ['Age', 'Number_of_Siblings_Spouses', 'Number_Of_Parents_Children', 'Fare']
    model = Classifier(attrs=attrs, numeric=nums, label='Survived')
    data_train = model.load_data('data/titanic/train.csv')
    data_test = model.load_data('data/titanic/test.csv')
    print('\n% titanic dataset train', np.shape(data_train), 'test', np.shape(data_test))
    return model, data_train, data_test


def anneal():
    attrs = ['family', 'product_type', 'steel', 'carbon', 'hardness', 'temper_rolling', 'condition', 'formability',
    'strength', 'non_ageing', 'surface_finish', 'surface_quality', 'enamelability', 'bc', 'bf', 'bt', 'bw_me', 'bl',
    'm', 'chrom', 'phos', 'cbond', 'marvi', 'exptl', 'ferro', 'corr', 'blue_bright_varn_clean', 'lustre', 'jurofm',
    's', 'p', 'shape', 'thick', 'width', 'len', 'oil', 'bore', 'packing']
    nums = ['thick', 'width', 'len']
    model = Classifier(attrs=attrs, numeric=nums, label='classes')
    data_train = model.load_data('data/anneal/anneal_train.csv')
    data_test = model.load_data('data/anneal/anneal_test.csv')
    print('\n% anneal dataset train', np.shape(data_train), 'test', np.shape(data_test))
    return model, data_train, data_test


def weight_lifting():
    attrs = ['new_window','num_window','roll_belt','pitch_belt','yaw_belt','total_accel_belt','kurtosis_roll_belt','kurtosis_picth_belt','kurtosis_yaw_belt','skewness_roll_belt','skewness_roll_belt','skewness_yaw_belt','max_roll_belt','max_picth_belt','max_yaw_belt','min_roll_belt','min_pitch_belt','min_yaw_belt','amplitude_roll_belt','amplitude_pitch_belt','amplitude_yaw_belt','var_total_accel_belt','avg_roll_belt','stddev_roll_belt','var_roll_belt','avg_pitch_belt','stddev_pitch_belt','var_pitch_belt','avg_yaw_belt','stddev_yaw_belt','var_yaw_belt','gyros_belt_x','gyros_belt_y','gyros_belt_z','accel_belt_x','accel_belt_y','accel_belt_z','magnet_belt_x','magnet_belt_y','magnet_belt_z','roll_arm','pitch_arm','yaw_arm','total_accel_arm','var_accel_arm','avg_roll_arm','stddev_roll_arm','var_roll_arm','avg_pitch_arm','stddev_pitch_arm','var_pitch_arm','avg_yaw_arm','stddev_yaw_arm','var_yaw_arm','gyros_arm_x','gyros_arm_y','gyros_arm_z','accel_arm_x','accel_arm_y','accel_arm_z','magnet_arm_x','magnet_arm_y','magnet_arm_z','kurtosis_roll_arm','kurtosis_picth_arm','kurtosis_yaw_arm','skewness_roll_arm','skewness_pitch_arm','skewness_yaw_arm','max_roll_arm','max_picth_arm','max_yaw_arm','min_roll_arm','min_pitch_arm','min_yaw_arm','amplitude_roll_arm','amplitude_pitch_arm','amplitude_yaw_arm','roll_dumbbell','pitch_dumbbell','yaw_dumbbell','kurtosis_roll_dumbbell','kurtosis_picth_dumbbell','kurtosis_yaw_dumbbell','skewness_roll_dumbbell','skewness_pitch_dumbbell','skewness_yaw_dumbbell','max_roll_dumbbell','max_picth_dumbbell','max_yaw_dumbbell','min_roll_dumbbell','min_pitch_dumbbell','min_yaw_dumbbell','amplitude_roll_dumbbell','amplitude_pitch_dumbbell','amplitude_yaw_dumbbell','total_accel_dumbbell','var_accel_dumbbell','avg_roll_dumbbell','stddev_roll_dumbbell','var_roll_dumbbell','avg_pitch_dumbbell','stddev_pitch_dumbbell','var_pitch_dumbbell','avg_yaw_dumbbell','stddev_yaw_dumbbell','var_yaw_dumbbell','gyros_dumbbell_x','gyros_dumbbell_y','gyros_dumbbell_z','accel_dumbbell_x','accel_dumbbell_y','accel_dumbbell_z','magnet_dumbbell_x','magnet_dumbbell_y','magnet_dumbbell_z','roll_forearm','pitch_forearm','yaw_forearm','kurtosis_roll_forearm','kurtosis_picth_forearm','kurtosis_yaw_forearm','skewness_roll_forearm','skewness_pitch_forearm','skewness_yaw_forearm','max_roll_forearm','max_picth_forearm','max_yaw_forearm','min_roll_forearm','min_pitch_forearm','min_yaw_forearm','amplitude_roll_forearm','amplitude_pitch_forearm','amplitude_yaw_forearm','total_accel_forearm','var_accel_forearm','avg_roll_forearm','stddev_roll_forearm','var_roll_forearm','avg_pitch_forearm','stddev_pitch_forearm','var_pitch_forearm','avg_yaw_forearm','stddev_yaw_forearm','var_yaw_forearm','gyros_forearm_x','gyros_forearm_y','gyros_forearm_z','accel_forearm_x','accel_forearm_y','accel_forearm_z','magnet_forearm_x','magnet_forearm_y','magnet_forearm_z']
    nums = ['num_window','roll_belt','pitch_belt','yaw_belt','total_accel_belt','kurtosis_roll_belt','kurtosis_picth_belt','kurtosis_yaw_belt','skewness_roll_belt','skewness_roll_belt','skewness_yaw_belt','max_roll_belt','max_picth_belt','max_yaw_belt','min_roll_belt','min_pitch_belt','min_yaw_belt','amplitude_roll_belt','amplitude_pitch_belt','amplitude_yaw_belt','var_total_accel_belt','avg_roll_belt','stddev_roll_belt','var_roll_belt','avg_pitch_belt','stddev_pitch_belt','var_pitch_belt','avg_yaw_belt','stddev_yaw_belt','var_yaw_belt','gyros_belt_x','gyros_belt_y','gyros_belt_z','accel_belt_x','accel_belt_y','accel_belt_z','magnet_belt_x','magnet_belt_y','magnet_belt_z','roll_arm','pitch_arm','yaw_arm','total_accel_arm','var_accel_arm','avg_roll_arm','stddev_roll_arm','var_roll_arm','avg_pitch_arm','stddev_pitch_arm','var_pitch_arm','avg_yaw_arm','stddev_yaw_arm','var_yaw_arm','gyros_arm_x','gyros_arm_y','gyros_arm_z','accel_arm_x','accel_arm_y','accel_arm_z','magnet_arm_x','magnet_arm_y','magnet_arm_z','kurtosis_roll_arm','kurtosis_picth_arm','kurtosis_yaw_arm','skewness_roll_arm','skewness_pitch_arm','skewness_yaw_arm','max_roll_arm','max_picth_arm','max_yaw_arm','min_roll_arm','min_pitch_arm','min_yaw_arm','amplitude_roll_arm','amplitude_pitch_arm','amplitude_yaw_arm','roll_dumbbell','pitch_dumbbell','yaw_dumbbell','kurtosis_roll_dumbbell','kurtosis_picth_dumbbell','kurtosis_yaw_dumbbell','skewness_roll_dumbbell','skewness_pitch_dumbbell','skewness_yaw_dumbbell','max_roll_dumbbell','max_picth_dumbbell','max_yaw_dumbbell','min_roll_dumbbell','min_pitch_dumbbell','min_yaw_dumbbell','amplitude_roll_dumbbell','amplitude_pitch_dumbbell','amplitude_yaw_dumbbell','total_accel_dumbbell','var_accel_dumbbell','avg_roll_dumbbell','stddev_roll_dumbbell','var_roll_dumbbell','avg_pitch_dumbbell','stddev_pitch_dumbbell','var_pitch_dumbbell','avg_yaw_dumbbell','stddev_yaw_dumbbell','var_yaw_dumbbell','gyros_dumbbell_x','gyros_dumbbell_y','gyros_dumbbell_z','accel_dumbbell_x','accel_dumbbell_y','accel_dumbbell_z','magnet_dumbbell_x','magnet_dumbbell_y','magnet_dumbbell_z','roll_forearm','pitch_forearm','yaw_forearm','kurtosis_roll_forearm','kurtosis_picth_forearm','kurtosis_yaw_forearm','skewness_roll_forearm','skewness_pitch_forearm','skewness_yaw_forearm','max_roll_forearm','max_picth_forearm','max_yaw_forearm','min_roll_forearm','min_pitch_forearm','min_yaw_forearm','amplitude_roll_forearm','amplitude_pitch_forearm','amplitude_yaw_forearm','total_accel_forearm','var_accel_forearm','avg_roll_forearm','stddev_roll_forearm','var_roll_forearm','avg_pitch_forearm','stddev_pitch_forearm','var_pitch_forearm','avg_yaw_forearm','stddev_yaw_forearm','var_yaw_forearm','gyros_forearm_x','gyros_forearm_y','gyros_forearm_z','accel_forearm_x','accel_forearm_y','accel_forearm_z','magnet_forearm_x','magnet_forearm_y','magnet_forearm_z']
    model = Classifier(attrs=attrs, numeric=nums, label='classe')
    data = model.load_data('data/weight_lifting/weight_lifting.csv')
    print('\n% weight lifting dataset', np.shape(data))
    return model, data


def yeast():
    attrs = ['sequence','mcg','gvh','alm','mit','erl','pox','vac','nuc']
    nums = ['mcg','gvh','alm','mit','erl','pox','vac','nuc']
    model = Classifier(attrs=attrs, numeric=nums, label='class')
    data = model.load_data('data/yeast/yeast.csv')
    print('\n% yeast dataset', np.shape(data))
    return model, data


def drug():
    attrs = ['Age','Gender','Education','Country','Ethnicity','Nscore','Escore','Oscore','Ascore','Cscore','Impulsive','SS']
    nums = attrs
    output = ['Alcohol','Amphet','Amyl','Benzos','Caff','Cannabis','Choc','Code','Crack','Ecstasy','Heroin','Ketamine','Legalh','LSD','Meth','Mushrooms','Nicotine','Semer','VSA']
    model = Classifier(attrs=attrs, numeric=nums, label=output[17])
    data = model.load_data('data/drug/drug.csv')
    print('\n% drug consumption dataset', np.shape(data))
    return model, data


def dry_bean():
    attrs = ['Area','Perimeter','MajorAxisLength','MinorAxisLength','AspectRation','Eccentricity','ConvexArea','EquivDiameter','Extent','Solidity','roundness','Compactness','ShapeFactor1','ShapeFactor2','ShapeFactor3','ShapeFactor4']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='Class')
    data = model.load_data('data/dry_bean/dry_bean.csv')
    print('\n% dry bean dataset', np.shape(data))
    return model, data


def eeg():
    attrs = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='eyeDetection')
    data = model.load_data('data/eeg/eeg.csv')
    print('\n% eeg dataset', np.shape(data))
    return model, data


def nursery():
    attrs = ['parents','has_nurs','form','children','housing','finance','social','health']
    nums = []
    model = Classifier(attrs=attrs, numeric=nums, label='class')
    data = model.load_data('data/nursery/nursery.csv')
    print('\n% nursery dataset', np.shape(data))
    return model, data


def intention():
    attrs = ['Administrative','Administrative_Duration','Informational','Informational_Duration','ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues','SpecialDay','Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend']
    nums = ['Administrative','Administrative_Duration','Informational','Informational_Duration','ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues','SpecialDay']
    model = Classifier(attrs=attrs, numeric=nums, label='Revenue')
    data = model.load_data('data/intention/intention.csv')
    print('\n% online shoppers intention dataset', np.shape(data))
    return model, data


def page_blocks():
    attrs = ['height','lenght','area','eccen','p_black','p_and','mean_tr','blackpix','blackand','wb_trans']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='class')
    data = model.load_data('data/page_blocks/page_blocks.csv')
    print('\n% page blocks dataset', np.shape(data))
    return model, data


def parkison():
    attrs = ['gender','PPE','DFA','RPDE','numPulses','numPeriodsPulses','meanPeriodPulses','stdDevPeriodPulses','locPctJitter','locAbsJitter','rapJitter','ppq5Jitter','ddpJitter','locShimmer','locDbShimmer','apq3Shimmer','apq5Shimmer','apq11Shimmer','ddaShimmer','meanAutoCorrHarmonicity','meanNoiseToHarmHarmonicity','meanHarmToNoiseHarmonicity','minIntensity','maxIntensity','meanIntensity','f1','f2','f3','f4','b1','b2','b3','b4','GQ_prc5_95','GQ_std_cycle_open','GQ_std_cycle_closed','GNE_mean','GNE_std','GNE_SNR_TKEO','GNE_SNR_SEO','GNE_NSR_TKEO','GNE_NSR_SEO','VFER_mean','VFER_std','VFER_entropy','VFER_SNR_TKEO','VFER_SNR_SEO','VFER_NSR_TKEO','VFER_NSR_SEO','IMF_SNR_SEO','IMF_SNR_TKEO','IMF_SNR_entropy','IMF_NSR_SEO','IMF_NSR_TKEO','IMF_NSR_entropy','mean_Log_energy','mean_MFCC_0th_coef','mean_MFCC_1st_coef','mean_MFCC_2nd_coef','mean_MFCC_3rd_coef','mean_MFCC_4th_coef','mean_MFCC_5th_coef','mean_MFCC_6th_coef','mean_MFCC_7th_coef','mean_MFCC_8th_coef','mean_MFCC_9th_coef','mean_MFCC_10th_coef','mean_MFCC_11th_coef','mean_MFCC_12th_coef','mean_delta_log_energy','mean_0th_delta','mean_1st_delta','mean_2nd_delta','mean_3rd_delta','mean_4th_delta','mean_5th_delta','mean_6th_delta','mean_7th_delta','mean_8th_delta','mean_9th_delta','mean_10th_delta','mean_11th_delta','mean_12th_delta','mean_delta_delta_log_energy','mean_delta_delta_0th','mean_1st_delta_delta','mean_2nd_delta_delta','mean_3rd_delta_delta','mean_4th_delta_delta','mean_5th_delta_delta','mean_6th_delta_delta','mean_7th_delta_delta','mean_8th_delta_delta','mean_9th_delta_delta','mean_10th_delta_delta','mean_11th_delta_delta','mean_12th_delta_delta','std_Log_energy','std_MFCC_0th_coef','std_MFCC_1st_coef','std_MFCC_2nd_coef','std_MFCC_3rd_coef','std_MFCC_4th_coef','std_MFCC_5th_coef','std_MFCC_6th_coef','std_MFCC_7th_coef','std_MFCC_8th_coef','std_MFCC_9th_coef','std_MFCC_10th_coef','std_MFCC_11th_coef','std_MFCC_12th_coef','std_delta_log_energy','std_0th_delta','std_1st_delta','std_2nd_delta','std_3rd_delta','std_4th_delta','std_5th_delta','std_6th_delta','std_7th_delta','std_8th_delta','std_9th_delta','std_10th_delta','std_11th_delta','std_12th_delta','std_delta_delta_log_energy','std_delta_delta_0th','std_1st_delta_delta','std_2nd_delta_delta','std_3rd_delta_delta','std_4th_delta_delta','std_5th_delta_delta','std_6th_delta_delta','std_7th_delta_delta','std_8th_delta_delta','std_9th_delta_delta','std_10th_delta_delta','std_11th_delta_delta','std_12th_delta_delta','Ea','Ed_1_coef','Ed_2_coef','Ed_3_coef','Ed_4_coef','Ed_5_coef','Ed_6_coef','Ed_7_coef','Ed_8_coef','Ed_9_coef','Ed_10_coef','det_entropy_shannon_1_coef','det_entropy_shannon_2_coef','det_entropy_shannon_3_coef','det_entropy_shannon_4_coef','det_entropy_shannon_5_coef','det_entropy_shannon_6_coef','det_entropy_shannon_7_coef','det_entropy_shannon_8_coef','det_entropy_shannon_9_coef','det_entropy_shannon_10_coef','det_entropy_log_1_coef','det_entropy_log_2_coef','det_entropy_log_3_coef','det_entropy_log_4_coef','det_entropy_log_5_coef','det_entropy_log_6_coef','det_entropy_log_7_coef','det_entropy_log_8_coef','det_entropy_log_9_coef','det_entropy_log_10_coef','det_TKEO_mean_1_coef','det_TKEO_mean_2_coef','det_TKEO_mean_3_coef','det_TKEO_mean_4_coef','det_TKEO_mean_5_coef','det_TKEO_mean_6_coef','det_TKEO_mean_7_coef','det_TKEO_mean_8_coef','det_TKEO_mean_9_coef','det_TKEO_mean_10_coef','det_TKEO_std_1_coef','det_TKEO_std_2_coef','det_TKEO_std_3_coef','det_TKEO_std_4_coef','det_TKEO_std_5_coef','det_TKEO_std_6_coef','det_TKEO_std_7_coef','det_TKEO_std_8_coef','det_TKEO_std_9_coef','det_TKEO_std_10_coef','app_entropy_shannon_1_coef','app_entropy_shannon_2_coef','app_entropy_shannon_3_coef','app_entropy_shannon_4_coef','app_entropy_shannon_5_coef','app_entropy_shannon_6_coef','app_entropy_shannon_7_coef','app_entropy_shannon_8_coef','app_entropy_shannon_9_coef','app_entropy_shannon_10_coef','app_entropy_log_1_coef','app_entropy_log_2_coef','app_entropy_log_3_coef','app_entropy_log_4_coef','app_entropy_log_5_coef','app_entropy_log_6_coef','app_entropy_log_7_coef','app_entropy_log_8_coef','app_entropy_log_9_coef','app_entropy_log_10_coef','app_det_TKEO_mean_1_coef','app_det_TKEO_mean_2_coef','app_det_TKEO_mean_3_coef','app_det_TKEO_mean_4_coef','app_det_TKEO_mean_5_coef','app_det_TKEO_mean_6_coef','app_det_TKEO_mean_7_coef','app_det_TKEO_mean_8_coef','app_det_TKEO_mean_9_coef','app_det_TKEO_mean_10_coef','app_TKEO_std_1_coef','app_TKEO_std_2_coef','app_TKEO_std_3_coef','app_TKEO_std_4_coef','app_TKEO_std_5_coef','app_TKEO_std_6_coef','app_TKEO_std_7_coef','app_TKEO_std_8_coef','app_TKEO_std_9_coef','app_TKEO_std_10_coef','Ea2','Ed2_1_coef','Ed2_2_coef','Ed2_3_coef','Ed2_4_coef','Ed2_5_coef','Ed2_6_coef','Ed2_7_coef','Ed2_8_coef','Ed2_9_coef','Ed2_10_coef','det_LT_entropy_shannon_1_coef','det_LT_entropy_shannon_2_coef','det_LT_entropy_shannon_3_coef','det_LT_entropy_shannon_4_coef','det_LT_entropy_shannon_5_coef','det_LT_entropy_shannon_6_coef','det_LT_entropy_shannon_7_coef','det_LT_entropy_shannon_8_coef','det_LT_entropy_shannon_9_coef','det_LT_entropy_shannon_10_coef','det_LT_entropy_log_1_coef','det_LT_entropy_log_2_coef','det_LT_entropy_log_3_coef','det_LT_entropy_log_4_coef','det_LT_entropy_log_5_coef','det_LT_entropy_log_6_coef','det_LT_entropy_log_7_coef','det_LT_entropy_log_8_coef','det_LT_entropy_log_9_coef','det_LT_entropy_log_10_coef','det_LT_TKEO_mean_1_coef','det_LT_TKEO_mean_2_coef','det_LT_TKEO_mean_3_coef','det_LT_TKEO_mean_4_coef','det_LT_TKEO_mean_5_coef','det_LT_TKEO_mean_6_coef','det_LT_TKEO_mean_7_coef','det_LT_TKEO_mean_8_coef','det_LT_TKEO_mean_9_coef','det_LT_TKEO_mean_10_coef','det_LT_TKEO_std_1_coef','det_LT_TKEO_std_2_coef','det_LT_TKEO_std_3_coef','det_LT_TKEO_std_4_coef','det_LT_TKEO_std_5_coef','det_LT_TKEO_std_6_coef','det_LT_TKEO_std_7_coef','det_LT_TKEO_std_8_coef','det_LT_TKEO_std_9_coef','det_LT_TKEO_std_10_coef','app_LT_entropy_shannon_1_coef','app_LT_entropy_shannon_2_coef','app_LT_entropy_shannon_3_coef','app_LT_entropy_shannon_4_coef','app_LT_entropy_shannon_5_coef','app_LT_entropy_shannon_6_coef','app_LT_entropy_shannon_7_coef','app_LT_entropy_shannon_8_coef','app_LT_entropy_shannon_9_coef','app_LT_entropy_shannon_10_coef','app_LT_entropy_log_1_coef','app_LT_entropy_log_2_coef','app_LT_entropy_log_3_coef','app_LT_entropy_log_4_coef','app_LT_entropy_log_5_coef','app_LT_entropy_log_6_coef','app_LT_entropy_log_7_coef','app_LT_entropy_log_8_coef','app_LT_entropy_log_9_coef','app_LT_entropy_log_10_coef','app_LT_TKEO_mean_1_coef','app_LT_TKEO_mean_2_coef','app_LT_TKEO_mean_3_coef','app_LT_TKEO_mean_4_coef','app_LT_TKEO_mean_5_coef','app_LT_TKEO_mean_6_coef','app_LT_TKEO_mean_7_coef','app_LT_TKEO_mean_8_coef','app_LT_TKEO_mean_9_coef','app_LT_TKEO_mean_10_coef','app_LT_TKEO_std_1_coef','app_LT_TKEO_std_2_coef','app_LT_TKEO_std_3_coef','app_LT_TKEO_std_4_coef','app_LT_TKEO_std_5_coef','app_LT_TKEO_std_6_coef','app_LT_TKEO_std_7_coef','app_LT_TKEO_std_8_coef','app_LT_TKEO_std_9_coef','app_LT_TKEO_std_10_coef','tqwt_energy_dec_1','tqwt_energy_dec_2','tqwt_energy_dec_3','tqwt_energy_dec_4','tqwt_energy_dec_5','tqwt_energy_dec_6','tqwt_energy_dec_7','tqwt_energy_dec_8','tqwt_energy_dec_9','tqwt_energy_dec_10','tqwt_energy_dec_11','tqwt_energy_dec_12','tqwt_energy_dec_13','tqwt_energy_dec_14','tqwt_energy_dec_15','tqwt_energy_dec_16','tqwt_energy_dec_17','tqwt_energy_dec_18','tqwt_energy_dec_19','tqwt_energy_dec_20','tqwt_energy_dec_21','tqwt_energy_dec_22','tqwt_energy_dec_23','tqwt_energy_dec_24','tqwt_energy_dec_25','tqwt_energy_dec_26','tqwt_energy_dec_27','tqwt_energy_dec_28','tqwt_energy_dec_29','tqwt_energy_dec_30','tqwt_energy_dec_31','tqwt_energy_dec_32','tqwt_energy_dec_33','tqwt_energy_dec_34','tqwt_energy_dec_35','tqwt_energy_dec_36','tqwt_entropy_shannon_dec_1','tqwt_entropy_shannon_dec_2','tqwt_entropy_shannon_dec_3','tqwt_entropy_shannon_dec_4','tqwt_entropy_shannon_dec_5','tqwt_entropy_shannon_dec_6','tqwt_entropy_shannon_dec_7','tqwt_entropy_shannon_dec_8','tqwt_entropy_shannon_dec_9','tqwt_entropy_shannon_dec_10','tqwt_entropy_shannon_dec_11','tqwt_entropy_shannon_dec_12','tqwt_entropy_shannon_dec_13','tqwt_entropy_shannon_dec_14','tqwt_entropy_shannon_dec_15','tqwt_entropy_shannon_dec_16','tqwt_entropy_shannon_dec_17','tqwt_entropy_shannon_dec_18','tqwt_entropy_shannon_dec_19','tqwt_entropy_shannon_dec_20','tqwt_entropy_shannon_dec_21','tqwt_entropy_shannon_dec_22','tqwt_entropy_shannon_dec_23','tqwt_entropy_shannon_dec_24','tqwt_entropy_shannon_dec_25','tqwt_entropy_shannon_dec_26','tqwt_entropy_shannon_dec_27','tqwt_entropy_shannon_dec_28','tqwt_entropy_shannon_dec_29','tqwt_entropy_shannon_dec_30','tqwt_entropy_shannon_dec_31','tqwt_entropy_shannon_dec_32','tqwt_entropy_shannon_dec_33','tqwt_entropy_shannon_dec_34','tqwt_entropy_shannon_dec_35','tqwt_entropy_shannon_dec_36','tqwt_entropy_log_dec_1','tqwt_entropy_log_dec_2','tqwt_entropy_log_dec_3','tqwt_entropy_log_dec_4','tqwt_entropy_log_dec_5','tqwt_entropy_log_dec_6','tqwt_entropy_log_dec_7','tqwt_entropy_log_dec_8','tqwt_entropy_log_dec_9','tqwt_entropy_log_dec_10','tqwt_entropy_log_dec_11','tqwt_entropy_log_dec_12','tqwt_entropy_log_dec_13','tqwt_entropy_log_dec_14','tqwt_entropy_log_dec_15','tqwt_entropy_log_dec_16','tqwt_entropy_log_dec_17','tqwt_entropy_log_dec_18','tqwt_entropy_log_dec_19','tqwt_entropy_log_dec_20','tqwt_entropy_log_dec_21','tqwt_entropy_log_dec_22','tqwt_entropy_log_dec_23','tqwt_entropy_log_dec_24','tqwt_entropy_log_dec_25','tqwt_entropy_log_dec_26','tqwt_entropy_log_dec_27','tqwt_entropy_log_dec_28','tqwt_entropy_log_dec_29','tqwt_entropy_log_dec_30','tqwt_entropy_log_dec_31','tqwt_entropy_log_dec_32','tqwt_entropy_log_dec_33','tqwt_entropy_log_dec_34','tqwt_entropy_log_dec_35','tqwt_entropy_log_dec_36','tqwt_TKEO_mean_dec_1','tqwt_TKEO_mean_dec_2','tqwt_TKEO_mean_dec_3','tqwt_TKEO_mean_dec_4','tqwt_TKEO_mean_dec_5','tqwt_TKEO_mean_dec_6','tqwt_TKEO_mean_dec_7','tqwt_TKEO_mean_dec_8','tqwt_TKEO_mean_dec_9','tqwt_TKEO_mean_dec_10','tqwt_TKEO_mean_dec_11','tqwt_TKEO_mean_dec_12','tqwt_TKEO_mean_dec_13','tqwt_TKEO_mean_dec_14','tqwt_TKEO_mean_dec_15','tqwt_TKEO_mean_dec_16','tqwt_TKEO_mean_dec_17','tqwt_TKEO_mean_dec_18','tqwt_TKEO_mean_dec_19','tqwt_TKEO_mean_dec_20','tqwt_TKEO_mean_dec_21','tqwt_TKEO_mean_dec_22','tqwt_TKEO_mean_dec_23','tqwt_TKEO_mean_dec_24','tqwt_TKEO_mean_dec_25','tqwt_TKEO_mean_dec_26','tqwt_TKEO_mean_dec_27','tqwt_TKEO_mean_dec_28','tqwt_TKEO_mean_dec_29','tqwt_TKEO_mean_dec_30','tqwt_TKEO_mean_dec_31','tqwt_TKEO_mean_dec_32','tqwt_TKEO_mean_dec_33','tqwt_TKEO_mean_dec_34','tqwt_TKEO_mean_dec_35','tqwt_TKEO_mean_dec_36','tqwt_TKEO_std_dec_1','tqwt_TKEO_std_dec_2','tqwt_TKEO_std_dec_3','tqwt_TKEO_std_dec_4','tqwt_TKEO_std_dec_5','tqwt_TKEO_std_dec_6','tqwt_TKEO_std_dec_7','tqwt_TKEO_std_dec_8','tqwt_TKEO_std_dec_9','tqwt_TKEO_std_dec_10','tqwt_TKEO_std_dec_11','tqwt_TKEO_std_dec_12','tqwt_TKEO_std_dec_13','tqwt_TKEO_std_dec_14','tqwt_TKEO_std_dec_15','tqwt_TKEO_std_dec_16','tqwt_TKEO_std_dec_17','tqwt_TKEO_std_dec_18','tqwt_TKEO_std_dec_19','tqwt_TKEO_std_dec_20','tqwt_TKEO_std_dec_21','tqwt_TKEO_std_dec_22','tqwt_TKEO_std_dec_23','tqwt_TKEO_std_dec_24','tqwt_TKEO_std_dec_25','tqwt_TKEO_std_dec_26','tqwt_TKEO_std_dec_27','tqwt_TKEO_std_dec_28','tqwt_TKEO_std_dec_29','tqwt_TKEO_std_dec_30','tqwt_TKEO_std_dec_31','tqwt_TKEO_std_dec_32','tqwt_TKEO_std_dec_33','tqwt_TKEO_std_dec_34','tqwt_TKEO_std_dec_35','tqwt_TKEO_std_dec_36','tqwt_medianValue_dec_1','tqwt_medianValue_dec_2','tqwt_medianValue_dec_3','tqwt_medianValue_dec_4','tqwt_medianValue_dec_5','tqwt_medianValue_dec_6','tqwt_medianValue_dec_7','tqwt_medianValue_dec_8','tqwt_medianValue_dec_9','tqwt_medianValue_dec_10','tqwt_medianValue_dec_11','tqwt_medianValue_dec_12','tqwt_medianValue_dec_13','tqwt_medianValue_dec_14','tqwt_medianValue_dec_15','tqwt_medianValue_dec_16','tqwt_medianValue_dec_17','tqwt_medianValue_dec_18','tqwt_medianValue_dec_19','tqwt_medianValue_dec_20','tqwt_medianValue_dec_21','tqwt_medianValue_dec_22','tqwt_medianValue_dec_23','tqwt_medianValue_dec_24','tqwt_medianValue_dec_25','tqwt_medianValue_dec_26','tqwt_medianValue_dec_27','tqwt_medianValue_dec_28','tqwt_medianValue_dec_29','tqwt_medianValue_dec_30','tqwt_medianValue_dec_31','tqwt_medianValue_dec_32','tqwt_medianValue_dec_33','tqwt_medianValue_dec_34','tqwt_medianValue_dec_35','tqwt_medianValue_dec_36','tqwt_meanValue_dec_1','tqwt_meanValue_dec_2','tqwt_meanValue_dec_3','tqwt_meanValue_dec_4','tqwt_meanValue_dec_5','tqwt_meanValue_dec_6','tqwt_meanValue_dec_7','tqwt_meanValue_dec_8','tqwt_meanValue_dec_9','tqwt_meanValue_dec_10','tqwt_meanValue_dec_11','tqwt_meanValue_dec_12','tqwt_meanValue_dec_13','tqwt_meanValue_dec_14','tqwt_meanValue_dec_15','tqwt_meanValue_dec_16','tqwt_meanValue_dec_17','tqwt_meanValue_dec_18','tqwt_meanValue_dec_19','tqwt_meanValue_dec_20','tqwt_meanValue_dec_21','tqwt_meanValue_dec_22','tqwt_meanValue_dec_23','tqwt_meanValue_dec_24','tqwt_meanValue_dec_25','tqwt_meanValue_dec_26','tqwt_meanValue_dec_27','tqwt_meanValue_dec_28','tqwt_meanValue_dec_29','tqwt_meanValue_dec_30','tqwt_meanValue_dec_31','tqwt_meanValue_dec_32','tqwt_meanValue_dec_33','tqwt_meanValue_dec_34','tqwt_meanValue_dec_35','tqwt_meanValue_dec_36','tqwt_stdValue_dec_1','tqwt_stdValue_dec_2','tqwt_stdValue_dec_3','tqwt_stdValue_dec_4','tqwt_stdValue_dec_5','tqwt_stdValue_dec_6','tqwt_stdValue_dec_7','tqwt_stdValue_dec_8','tqwt_stdValue_dec_9','tqwt_stdValue_dec_10','tqwt_stdValue_dec_11','tqwt_stdValue_dec_12','tqwt_stdValue_dec_13','tqwt_stdValue_dec_14','tqwt_stdValue_dec_15','tqwt_stdValue_dec_16','tqwt_stdValue_dec_17','tqwt_stdValue_dec_18','tqwt_stdValue_dec_19','tqwt_stdValue_dec_20','tqwt_stdValue_dec_21','tqwt_stdValue_dec_22','tqwt_stdValue_dec_23','tqwt_stdValue_dec_24','tqwt_stdValue_dec_25','tqwt_stdValue_dec_26','tqwt_stdValue_dec_27','tqwt_stdValue_dec_28','tqwt_stdValue_dec_29','tqwt_stdValue_dec_30','tqwt_stdValue_dec_31','tqwt_stdValue_dec_32','tqwt_stdValue_dec_33','tqwt_stdValue_dec_34','tqwt_stdValue_dec_35','tqwt_stdValue_dec_36','tqwt_minValue_dec_1','tqwt_minValue_dec_2','tqwt_minValue_dec_3','tqwt_minValue_dec_4','tqwt_minValue_dec_5','tqwt_minValue_dec_6','tqwt_minValue_dec_7','tqwt_minValue_dec_8','tqwt_minValue_dec_9','tqwt_minValue_dec_10','tqwt_minValue_dec_11','tqwt_minValue_dec_12','tqwt_minValue_dec_13','tqwt_minValue_dec_14','tqwt_minValue_dec_15','tqwt_minValue_dec_16','tqwt_minValue_dec_17','tqwt_minValue_dec_18','tqwt_minValue_dec_19','tqwt_minValue_dec_20','tqwt_minValue_dec_21','tqwt_minValue_dec_22','tqwt_minValue_dec_23','tqwt_minValue_dec_24','tqwt_minValue_dec_25','tqwt_minValue_dec_26','tqwt_minValue_dec_27','tqwt_minValue_dec_28','tqwt_minValue_dec_29','tqwt_minValue_dec_30','tqwt_minValue_dec_31','tqwt_minValue_dec_32','tqwt_minValue_dec_33','tqwt_minValue_dec_34','tqwt_minValue_dec_35','tqwt_minValue_dec_36','tqwt_maxValue_dec_1','tqwt_maxValue_dec_2','tqwt_maxValue_dec_3','tqwt_maxValue_dec_4','tqwt_maxValue_dec_5','tqwt_maxValue_dec_6','tqwt_maxValue_dec_7','tqwt_maxValue_dec_8','tqwt_maxValue_dec_9','tqwt_maxValue_dec_10','tqwt_maxValue_dec_11','tqwt_maxValue_dec_12','tqwt_maxValue_dec_13','tqwt_maxValue_dec_14','tqwt_maxValue_dec_15','tqwt_maxValue_dec_16','tqwt_maxValue_dec_17','tqwt_maxValue_dec_18','tqwt_maxValue_dec_19','tqwt_maxValue_dec_20','tqwt_maxValue_dec_21','tqwt_maxValue_dec_22','tqwt_maxValue_dec_23','tqwt_maxValue_dec_24','tqwt_maxValue_dec_25','tqwt_maxValue_dec_26','tqwt_maxValue_dec_27','tqwt_maxValue_dec_28','tqwt_maxValue_dec_29','tqwt_maxValue_dec_30','tqwt_maxValue_dec_31','tqwt_maxValue_dec_32','tqwt_maxValue_dec_33','tqwt_maxValue_dec_34','tqwt_maxValue_dec_35','tqwt_maxValue_dec_36','tqwt_skewnessValue_dec_1','tqwt_skewnessValue_dec_2','tqwt_skewnessValue_dec_3','tqwt_skewnessValue_dec_4','tqwt_skewnessValue_dec_5','tqwt_skewnessValue_dec_6','tqwt_skewnessValue_dec_7','tqwt_skewnessValue_dec_8','tqwt_skewnessValue_dec_9','tqwt_skewnessValue_dec_10','tqwt_skewnessValue_dec_11','tqwt_skewnessValue_dec_12','tqwt_skewnessValue_dec_13','tqwt_skewnessValue_dec_14','tqwt_skewnessValue_dec_15','tqwt_skewnessValue_dec_16','tqwt_skewnessValue_dec_17','tqwt_skewnessValue_dec_18','tqwt_skewnessValue_dec_19','tqwt_skewnessValue_dec_20','tqwt_skewnessValue_dec_21','tqwt_skewnessValue_dec_22','tqwt_skewnessValue_dec_23','tqwt_skewnessValue_dec_24','tqwt_skewnessValue_dec_25','tqwt_skewnessValue_dec_26','tqwt_skewnessValue_dec_27','tqwt_skewnessValue_dec_28','tqwt_skewnessValue_dec_29','tqwt_skewnessValue_dec_30','tqwt_skewnessValue_dec_31','tqwt_skewnessValue_dec_32','tqwt_skewnessValue_dec_33','tqwt_skewnessValue_dec_34','tqwt_skewnessValue_dec_35','tqwt_skewnessValue_dec_36','tqwt_kurtosisValue_dec_1','tqwt_kurtosisValue_dec_2','tqwt_kurtosisValue_dec_3','tqwt_kurtosisValue_dec_4','tqwt_kurtosisValue_dec_5','tqwt_kurtosisValue_dec_6','tqwt_kurtosisValue_dec_7','tqwt_kurtosisValue_dec_8','tqwt_kurtosisValue_dec_9','tqwt_kurtosisValue_dec_10','tqwt_kurtosisValue_dec_11','tqwt_kurtosisValue_dec_12','tqwt_kurtosisValue_dec_13','tqwt_kurtosisValue_dec_14','tqwt_kurtosisValue_dec_15','tqwt_kurtosisValue_dec_16','tqwt_kurtosisValue_dec_17','tqwt_kurtosisValue_dec_18','tqwt_kurtosisValue_dec_19','tqwt_kurtosisValue_dec_20','tqwt_kurtosisValue_dec_21','tqwt_kurtosisValue_dec_22','tqwt_kurtosisValue_dec_23','tqwt_kurtosisValue_dec_24','tqwt_kurtosisValue_dec_25','tqwt_kurtosisValue_dec_26','tqwt_kurtosisValue_dec_27','tqwt_kurtosisValue_dec_28','tqwt_kurtosisValue_dec_29','tqwt_kurtosisValue_dec_30','tqwt_kurtosisValue_dec_31','tqwt_kurtosisValue_dec_32','tqwt_kurtosisValue_dec_33','tqwt_kurtosisValue_dec_34','tqwt_kurtosisValue_dec_35','tqwt_kurtosisValue_dec_36']
    nums = attrs[1:]
    model = Classifier(attrs=attrs, numeric=nums, label='class')
    data = model.load_data('data/parkison_disease/parkison_disease.csv')
    print('\n% parkison disease dataset', np.shape(data))
    return model, data


def pendigits():
    attrs = ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12','a13','a14','a15','a16']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='class')
    data_train = model.load_data('data/pendigits/train.csv')
    data_test = model.load_data('data/pendigits/test.csv')
    print('\n% pendigits train dataset', np.shape(data_train), 'test', np.shape(data_test))
    return model, data_train, data_test


def wall_robot():
    attrs = ['US1','US2','US3','US4','US5','US6','US7','US8','US9','US10','US11','US12','US13','US14','US15','US16','US17','US18','US19','US20','US21','US22','US23','US24']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='Class')
    data = model.load_data('data/wall_following_robot/wall_following_robot.csv')
    print('\n% wall_following_robot dataset', np.shape(data))
    return model, data


def glass():
    attrs = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']
    nums = attrs
    model = Classifier(attrs=attrs, numeric=nums, label='Type')
    data = model.load_data('data/glass/glass.csv')
    print('\n% glass dataset', np.shape(data))
    return model, data


def flags():
    attrs = ['name','landmass','zone','area','population','language','bars','stripes','colours','red','green','blue','gold','white','black','orange','mainhue','circles','crosses','saltires','quarters','sunstars','crescent','triangle','icon','animate','text','topleft','botright']
    nums = ['area','population','stripes','colours','sunstars']
    model = Classifier(attrs=attrs, numeric=nums, label='religion')
    data = model.load_data('data/flags/flags.csv')
    print('\n% flags dataset', np.shape(data))
    return model, data

"""
PHASE 1-3: DATA LOADING, CLEANING, EDA & FEATURE ENGINEERING
Credit Card Default Prediction Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: DATA LOADING & CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_clean_data(filepath):
    """
    Load data from CSV and perform initial cleaning
    
    Parameters:
    -----------
    filepath : str
        Path to the data CSV file
    
    Returns:
    --------
    df : pd.DataFrame
        Cleaned dataframe
    """
    
    print("="*80)
    print("PHASE 1: DATA LOADING & CLEANING")
    print("="*80)
    
    # Load data (header is on row 1, not row 0)
    df = pd.read_csv(filepath, header=1)
    
    print(f"\n✓ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"✓ Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    # Rename target column
    df.rename(columns={'default payment next month': 'DEFAULT'}, inplace=True)
    
    print(f"✓ Renamed target column to 'DEFAULT'")
    
    # Fix EDUCATION values
    education_invalid = df['EDUCATION'].value_counts().get(0, 0) + \
                       df['EDUCATION'].value_counts().get(5, 0) + \
                       df['EDUCATION'].value_counts().get(6, 0)
    
    df['EDUCATION'] = df['EDUCATION'].replace([0, 5, 6], 4)
    print(f"✓ Fixed {education_invalid} invalid EDUCATION values (0,5,6 → 4)")
    
    # Fix MARRIAGE values
    marriage_invalid = df['MARRIAGE'].value_counts().get(0, 0)
    df['MARRIAGE'] = df['MARRIAGE'].replace(0, 3)
    print(f"✓ Fixed {marriage_invalid} invalid MARRIAGE values (0 → 3)")
    
    # Drop ID column (not useful)
    df.drop('ID', axis=1, inplace=True)
    print(f"✓ Dropped ID column (not useful for modeling)")
    
    # Rename payment columns for clarity
    rename_dict = {
        'PAY_0': 'PAY_SEPT',
        'PAY_2': 'PAY_AUG',
        'PAY_3': 'PAY_JUL',
        'PAY_4': 'PAY_JUN',
        'PAY_5': 'PAY_MAY',
        'PAY_6': 'PAY_APR'
    }
    df.rename(columns=rename_dict, inplace=True)
    print(f"✓ Renamed payment columns (PAY_0 → PAY_SEPT, etc.)")
    
    # Check for nulls and duplicates
    nulls = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    print(f"\n✓ Null values: {nulls}")
    print(f"✓ Duplicate rows: {duplicates}")
    
    print(f"\n✅ Data cleaning complete!")
    print(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def exploratory_data_analysis(df, output_path=None):
    """
    Perform comprehensive EDA on the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    output_path : str, optional
        Path to save visualizations
    """
    
    print("\n" + "="*80)
    print("PHASE 2: EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # Summary statistics
    print("\n📊 DATASET SUMMARY:")
    print(f"Shape: {df.shape}")
    print(f"\nDefault rate: {(df['DEFAULT']==1).sum()/len(df)*100:.1f}%")
    print(f"  No default: {(df['DEFAULT']==0).sum():,} ({(df['DEFAULT']==0).sum()/len(df)*100:.1f}%)")
    print(f"  Default: {(df['DEFAULT']==1).sum():,} ({(df['DEFAULT']==1).sum()/len(df)*100:.1f}%)")
    
    # Correlation with DEFAULT
    print("\n📈 CORRELATIONS WITH DEFAULT (Top 15):")
    corr_with_default = df.corr()['DEFAULT'].sort_values(ascending=False)
    print(corr_with_default[corr_with_default.index != 'DEFAULT'].head(15))
    
    # Data types
    print("\n📋 DATA TYPES:")
    print(df.dtypes.value_counts())
    
    return corr_with_default


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(df):
    """
    Create new features from existing ones
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with engineered features
    """
    
    print("\n" + "="*80)
    print("PHASE 3: FEATURE ENGINEERING")
    print("="*80)
    
    # Define column groups
    pay_stat_cols = ['PAY_SEPT', 'PAY_AUG', 'PAY_JUL', 'PAY_JUN', 'PAY_MAY', 'PAY_APR']
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    pay_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    print(f"\n🔨 Creating 11 new features...")
    
    # 1. Utilization Ratio (Bill / Credit Limit)
    df['UTIL_RATIO'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1)
    print("  1. UTIL_RATIO = BILL_AMT1 / (LIMIT_BAL + 1)")
    
    # 2. Payment Ratio (Payment / Bill)
    df['PAY_RATIO_1'] = df['PAY_AMT1'] / (df['BILL_AMT1'] + 1)
    print("  2. PAY_RATIO_1 = PAY_AMT1 / (BILL_AMT1 + 1)")
    
    # 3. Months with Delay
    df['MONTHS_DELAYED'] = (df[pay_stat_cols] > 0).sum(axis=1)
    print("  3. MONTHS_DELAYED = count of months with payment delay")
    
    # 4. Maximum Delay
    df['MAX_DELAY'] = df[pay_stat_cols].max(axis=1)
    print("  4. MAX_DELAY = worst delay in 6 months")
    
    # 5. Average Bill Amount
    df['AVG_BILL_AMT'] = df[bill_cols].mean(axis=1)
    print("  5. AVG_BILL_AMT = average of 6-month bills")
    
    # 6. Average Payment Amount
    df['AVG_PAY_AMT'] = df[pay_cols].mean(axis=1)
    print("  6. AVG_PAY_AMT = average of 6-month payments")
    
    # 7. Total Paid
    df['TOTAL_PAID'] = df[pay_cols].sum(axis=1)
    print("  7. TOTAL_PAID = sum of 6-month payments")
    
    # 8. Total Bill
    df['TOTAL_BILL'] = df[bill_cols].sum(axis=1)
    print("  8. TOTAL_BILL = sum of 6-month bills")
    
    # 9. Overall Payment Ratio
    df['OVERALL_PAY_RATIO'] = df['TOTAL_PAID'] / (df['TOTAL_BILL'] + 1)
    print("  9. OVERALL_PAY_RATIO = TOTAL_PAID / TOTAL_BILL")
    
    # 10. Bill Trend (Is debt growing?)
    df['BILL_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']
    print("  10. BILL_TREND = BILL_AMT1 - BILL_AMT6 (recent minus oldest)")
    
    # 11. Revolving Utilization (clipped at 2.0)
    df['REVOLVING_UTIL'] = (df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1)).clip(0, 2)
    print("  11. REVOLVING_UTIL = UTIL_RATIO clipped at 2.0")
    
    print(f"\n✅ Feature engineering complete! New shape: {df.shape}")
    
    return df, pay_stat_cols, bill_cols, pay_cols


def apply_log_transformations(df, bill_cols, pay_cols):
    """
    Apply log transformation to skewed columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    bill_cols : list
        Bill amount columns
    pay_cols : list
        Payment amount columns
    
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with log-transformed features
    """
    
    print("\n" + "="*80)
    print("APPLYING LOG TRANSFORMATIONS")
    print("="*80)
    
    skewed = bill_cols + pay_cols + ['LIMIT_BAL', 'AVG_BILL_AMT', 'AVG_PAY_AMT', 
                                      'TOTAL_PAID', 'TOTAL_BILL']
    
    print(f"\nApplying log1p to {len(skewed)} skewed columns...")
    for col in skewed:
        df[col + '_LOG'] = np.log1p(df[col].clip(lower=0))
    
    print(f"✅ Created {len(skewed)} log-transformed columns")
    
    return df


def select_features(df, corr_threshold=0.05, multicollinear_threshold=0.85):
    """
    3-stage feature selection pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    corr_threshold : float
        Minimum correlation with DEFAULT
    multicollinear_threshold : float
        Maximum inter-feature correlation
    
    Returns:
    --------
    final_features : list
        Selected features
    """
    
    print("\n" + "="*80)
    print("3-STAGE FEATURE SELECTION")
    print("="*80)
    
    X = df.drop('DEFAULT', axis=1)
    y = df['DEFAULT']
    
    # STAGE 1: Correlation Filter
    print("\n🔵 STAGE 1: Correlation Filter")
    corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
    stage1_features = corr_with_target[corr_with_target > corr_threshold].index.tolist()
    
    print(f"  Input features: {X.shape[1]}")
    print(f"  Passed filter (|r| > {corr_threshold}): {len(stage1_features)}")
    print(f"  Features removed: {X.shape[1] - len(stage1_features)}")
    
    # STAGE 2: Remove Multicollinearity
    print("\n🔵 STAGE 2: Remove Multicollinearity")
    X_stage1 = X[stage1_features]
    corr_matrix = X_stage1.corr().abs()
    
    # Find highly correlated pairs
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > multicollinear_threshold:
                col = corr_matrix.columns[j]
                to_drop.add(col)
    
    stage2_features = [f for f in stage1_features if f not in to_drop]
    
    print(f"  Input features: {len(stage1_features)}")
    print(f"  Dropped (corr > {multicollinear_threshold}): {len(to_drop)}")
    print(f"  Remaining: {len(stage2_features)}")
    print(f"  Dropped features: {sorted(to_drop)[:5]}... ({len(to_drop)} total)")
    
    # STAGE 3: Random Forest Importance
    print("\n🔵 STAGE 3: Random Forest Importance Ranking")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    X_stage2 = X[stage2_features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_stage2)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    
    importances = pd.DataFrame({
        'feature': stage2_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"  Evaluated: {len(stage2_features)} features")
    print(f"\n  Top 10 features by importance:")
    for idx, row in importances.head(10).iterrows():
        print(f"    {row['feature']:20s} {row['importance']:.4f}")
    
    final_features = importances['feature'].tolist()
    
    print(f"\n  ✅ Final feature count: {len(final_features)}")
    
    return final_features


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # Phase 1: Load and clean
    df = load_and_clean_data("default_of_credit_card_clients_xls_-_Data.csv")
    
    # Phase 2: EDA
    corr_target = exploratory_data_analysis(df)
    
    # Phase 3: Feature Engineering
    df, pay_stat_cols, bill_cols, pay_cols = engineer_features(df)
    df = apply_log_transformations(df, bill_cols, pay_cols)
    
    # Feature Selection
    final_features = select_features(df)
    
    # Save for next phases
    import json
    with open('final_features.json', 'w') as f:
        json.dump(final_features, f)
    
    df.to_csv('data_engineered.csv', index=False)
    
    print("\n" + "="*80)
    print("✅ PHASES 1-3 COMPLETE!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - final_features.json (selected features for modeling)")
    print(f"  - data_engineered.csv (engineered dataset)")

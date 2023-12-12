import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, floor, when, isnan
from pyspark.sql.types import IntegerType, FloatType, ByteType
from pyspark.sql import functions as F
from ..utils import setup_logger, setup_timer
import argparse

logger = setup_logger("Prallel Preprocessing")
timer = setup_timer(logger)

spark = SparkSession.builder \
    .appName("Prallel Processing") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

@timer
def readData(path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    #test_df = pd.read_csv(test_path)
    return df

def floorify(x, lo):
    """example: x in [0, 0.01] -> x := 0"""
    if x is None:
        return -1  
    return lo if x <= lo + 0.01 and x >= lo else x

floorify_udf = udf(floorify, FloatType())

def floorify_zeros(df, column):
    """look around values [0,0.01] and determine if in proximity it's categorical. If yes - floorify"""
    return df.withColumn(column, when((col(column) >= 0) & (col(column) <= 0.01), 0).otherwise(col(column)))


def floorify_ones(df, column):
    """look around values [1,1.01] and determine if in proximity it's categorical. If yes - floorify"""    
    return df.withColumn(column, when((col(column) >= 1) & (col(column) <= 1.01), 1).otherwise(col(column)))


def convert_na(df, column):
    """nan -> -1 if positive values"""
    return df.withColumn(column, when(isnan(col(column)), -1).otherwise(col(column)))


def convert_to_int(df, column):
    """float -> int8 if possible"""
    return df.withColumn(column, col(column).cast(ByteType()))


def floorify_ones_and_zeros(df, column):
    """do everything"""
    df = floorify_zeros(df, column)
    df = floorify_ones(df, column)
    df = convert_to_int(df, column)
    return df

floorify_ones_and_zeros_udf = udf(floorify_ones_and_zeros, FloatType())

def floorify_frac_udf(interval):
    def floorify_frac(x):
        return int(x // interval) if x is not None else -1
    return F.udf(floorify_frac, IntegerType())

def apply_floorify_frac(df, column, interval=1.0, additional=0.0):
    if additional != 0.0:
        df = df.withColumn(column, F.col(column) + F.lit(additional))
    return df.withColumn(column, floorify_frac_udf(interval)(F.col(column)))


@timer
def processing1_feat(df): 
    df = apply_floorify_frac(df, 'B_4', 1/78)
    df = apply_floorify_frac(df, 'B_16', 1/12)
    df = apply_floorify_frac(df, 'B_20', 1/17)
    df = apply_floorify_frac(df, 'B_22', 1/2)
    df = apply_floorify_frac(df, 'B_30')
    df = apply_floorify_frac(df, 'B_31')
    df = apply_floorify_frac(df, 'B_32')
    df = apply_floorify_frac(df, 'B_33')
    df = apply_floorify_frac(df, 'B_38')
    df = apply_floorify_frac(df, 'B_41')
    df = apply_floorify_frac(df, 'D_39', 1/34)
    df = apply_floorify_frac(df, 'D_44', 1/8)
    df = apply_floorify_frac(df, 'D_49', 1/71)
    df = apply_floorify_frac(df, 'D_51', 1/3)
    df = apply_floorify_frac(df, 'D_59', 1/48, additional=5/48)  # Special case with additional adjustment
    df = apply_floorify_frac(df, 'D_65', 1/38)
    df = apply_floorify_frac(df, 'D_66')
    df = apply_floorify_frac(df, 'D_68')
    df = apply_floorify_frac(df, 'D_70', 1/4)
    df = apply_floorify_frac(df, 'D_72', 1/3)
    df = apply_floorify_frac(df, 'D_74', 1/14)
    df = apply_floorify_frac(df, 'D_75', 1/15)
    df = apply_floorify_frac(df, 'D_78', 1/2)
    df = apply_floorify_frac(df, 'D_79', 1/2)
    df = apply_floorify_frac(df, 'D_80', 1/5)
    df = apply_floorify_frac(df, 'D_81')
    df = apply_floorify_frac(df, 'D_82', 1/2)
    df = apply_floorify_frac(df, 'D_83')
    df = apply_floorify_frac(df, 'D_84', 1/2)
    df = apply_floorify_frac(df, 'D_86')
    df = apply_floorify_frac(df, 'D_87')
    df = apply_floorify_frac(df, 'D_89', 1/9)
    df = apply_floorify_frac(df, 'D_91', 1/2)
    df = apply_floorify_frac(df, 'D_92')
    df = apply_floorify_frac(df, 'D_93')
    df = apply_floorify_frac(df, 'D_94')
    df = apply_floorify_frac(df, 'D_96')
    df = apply_floorify_frac(df, 'D_103')
    df = apply_floorify_frac(df, 'D_106', 1/23)
    df = apply_floorify_frac(df, 'D_107', 1/3)
    df = apply_floorify_frac(df, 'D_108')
    df = apply_floorify_frac(df, 'D_109')
    df = apply_floorify_frac(df, 'D_111', 1/2)
    df = apply_floorify_frac(df, 'D_113', 1/5)
    df = apply_floorify_frac(df, 'D_114')
    df = apply_floorify_frac(df, 'D_116')
    df = apply_floorify_frac(df, 'D_117', 1, additional=1)  # Special case with additional adjustment
    df = apply_floorify_frac(df, 'D_120')
    df = apply_floorify_frac(df, 'D_122', 1/7)
    df = apply_floorify_frac(df, 'D_123')
    df = apply_floorify_frac(df, 'D_124', 1/22, additional=1/22)  # Special case with additional adjustment
    df = apply_floorify_frac(df, 'D_125')
    df = apply_floorify_frac(df, 'D_126', 1, additional=1)  # Special case with additional adjustment
    df = apply_floorify_frac(df, 'D_127')
    df = apply_floorify_frac(df, 'D_129')
    df = apply_floorify_frac(df, 'D_135')
    df = apply_floorify_frac(df, 'D_136', 1/4)
    df = apply_floorify_frac(df, 'D_137')
    df = apply_floorify_frac(df, 'D_138', 1/2)
    df = apply_floorify_frac(df, 'D_139')
    df = apply_floorify_frac(df, 'D_140')
    df = apply_floorify_frac(df, 'D_143')
    df = apply_floorify_frac(df, 'D_145', 1/11)
    df = apply_floorify_frac(df, 'R_2')
    df = apply_floorify_frac(df, 'R_3', 1/10)
    df = apply_floorify_frac(df,'R_4')
    df = apply_floorify_frac(df,'R_5',1/2)
    df = apply_floorify_frac(df, 'R_8')
    df = apply_floorify_frac(df, 'R_9',1/6)
    df = apply_floorify_frac(df, 'R_10')
    df = apply_floorify_frac(df, 'R_11',1/2)
    df = apply_floorify_frac(df, 'R_13',1/31)
    df = apply_floorify_frac(df, 'R_15')
    df = apply_floorify_frac(df, 'R_16',1/2)
    df = apply_floorify_frac(df, 'R_17' ,1/35)
    df = apply_floorify_frac(df,'R_18',1/31)
    df = apply_floorify_frac(df, 'R_19')
    df = apply_floorify_frac(df, 'R_20')
    df = apply_floorify_frac(df, 'R_21')
    df = apply_floorify_frac(df, 'R_22')
    df = apply_floorify_frac(df,'R_23')
    df = apply_floorify_frac(df, 'R_24')
    df = apply_floorify_frac(df, 'R_25')
    df = apply_floorify_frac(df,'R_26',1/28)
    df = apply_floorify_frac(df, 'R_28')
    df = apply_floorify_frac(df,'S_6')
    df = apply_floorify_frac(df, 'S_11',+5/25, additional=1/25)
    df = apply_floorify_frac(df,'S_15',+3/10, additional=1/10)
    df = apply_floorify_frac(df,'S_18')
    df = apply_floorify_frac(df,'S_20')
    df = df.withColumn('B_19', (F.floor(df['B_19'] * 100)).cast(IntegerType()))

    return df

@timer
def processing2_feat(df):
    # this one has many more value overlaps, but the splits can be identified by S_15
    conditions = [
        (df.S_8.between(0.30, 0.35) & (df.S_15 <= 6), 0.3224889650033656),
        (df.S_8.between(0.30, 0.35) & (df.S_15 == 7), 0.3145925513763017),
        (df.S_8.between(0.45, 0.477) & (df.S_15 == 3), 0.4570436553944634),
        (df.S_8.between(0.45, 0.477) & (df.S_15 == 5), 0.4636765662005172),
        (df.S_8.between(0.45, 0.477) & (df.S_15 == 6), 0.4592546209653157),
        (df.S_8.between(0.55, 0.65) & (df.S_15 == 5), 0.5938092592144236),
        (df.S_8.between(0.55, 0.65) & (df.S_15 == 4), 0.5994946974629933),
        (df.S_8.between(0.55, 0.65) & (df.S_15 <= 2), 0.6017056828901041),
        (df.S_8.between(0.73, 0.78) & (df.S_15 == 3), 0.7441567340107059),
        (df.S_8.between(0.73, 0.78) & (df.S_15 == 5), 0.7517372106519937),
        (df.S_8.between(0.73, 0.78) & (df.S_15 == 4), 0.7586861099807893),
        (df.S_8.between(0.91, 0.98) & (df.S_15 == 4), 0.9147189165383852),
        (df.S_8.between(0.91, 0.98) & (df.S_15 <= 2), 0.9327230426634736),
        (df.S_8.between(0.91, 0.98) & (df.S_15 == 3), 0.935565546481781),
        (df.S_8.between(1.12, 1.17) & (df.S_15 <= 2), 1.1440303975988897),
        (df.S_8.between(1.12, 1.17) & (df.S_15 == 3), 1.151926881019957)
    ]

    for condition, value in conditions:
        df = df.withColumn('S_8_updated', F.when(condition, value).otherwise(F.col('S_8')))
    # Replace the old column with the new one
    df = df.drop('S_8').withColumnRenamed('S_8_updated', 'S_8')
    
    floor_vals = (0, 0.1017056275625063, 0.119709415455368, 0.1667719530078215, 0.2438408100936861, 
              0.3578648754166172, 0.4055590769093041, 0.4772583808904347, 0.4876816287061991, 
              0.6620341135675392, 0.7005685574395781, 0.8509160456526623, 1, 1.0145299163657109, 
              1.1051803467580654, 1.2214158871037435)
    
    for c in floor_vals:
        df = df.withColumn('S_8', floorify_udf(df.S_8, F.lit(c)))
    df = df.withColumn('S_8', (F.round(df.S_8 * 3166)).cast('int'))
    logger.info('processing2_feat done')      
    return df

@timer
def processing3_feat(df):
    float_cols = [col for col, dtype in df.dtypes if dtype == 'float']

    for col in float_cols:
        df = df.withColumn(col, floorify_ones_and_zeros_udf(df[col]))
        
    logger.info('processing3_feat done')      
    return df

@timer
def write_parquet(df, path):
    df.write.parquet(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--output_path", type=str, default="parquet/train.parquet")
    args = parser.parse_args()
    
    # * Load Data
    data = readData(args.train_path)
    
    #preprocissing 
    data = processing1_feat(data)
    data = processing2_feat(data)
    data = processing3_feat(data)
    
    #write to parquet
    write_parquet(data,args.output_path)
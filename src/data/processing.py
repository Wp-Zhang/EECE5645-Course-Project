import numpy as np
import pandas as pd 

import pyspark
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, types
from pyspark.sql import SQLContext
import findspark
from pyspark import SparkContext


# Initialize Spark
def init_spark():
    findspark.init()
    #sc = SparkContext(appName='ML_spark')
    return SparkSession.builder.master("local[*]").getOrCreate()


# Create a Spark schema based on the types of columns
def create_spark_schema(series, string_dtypes, date_dtypes, types_map):
    fields = []
    for index, value in series.items():
        dtype = types_map.get(str(value), types.StringType())
        nullable = True
        if index in string_dtypes:
            dtype = types.StringType()
        elif index in date_dtypes:
            dtype = types.DateType()
        fields.append(types.StructField(index, dtype, nullable))
    return types.StructType(fields)

# check data types
def print_splits(*msg):
    for m in msg:
        print(m)
        print()


def main():
    train_path = "/scratch/luo.min/Project/train_data.csv"
    label_path = "/scratch/luo.min/Project/train_labels.csv"
    test_path = "/scratch/luo.min/Project/test_data.csv"

    # Initialize Spark
    spark = init_spark()

    #starting from the small subset 
    train_df = pd.read_csv(train_path, nrows=100)
    test_df = pd.read_csv(test_path, nrows=100)
    label_df = pd.read_csv(label_path, nrows=100)

    train_types = train_df.dtypes
    train_types_count = train_types.value_counts()

    ## Test types
    test_types = test_df.dtypes
    test_types_count = test_types.value_counts()

    ## Label types
    label_types = label_df.dtypes
    label_types_count = label_types.value_counts()

    print_splits(train_types_count, test_types_count, label_types_count)

    # Types mapper
    types_map = {
        "object": types.StringType(),
        "float64": types.FloatType(),
        "int64": types.IntegerType(),}

    # Known dtypes
    string_dtypes = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    date_dtypes = ['S_2']


    train_schema = create_spark_schema(train_types, string_dtypes, date_dtypes, types_map)
    test_schema = create_spark_schema(test_types,string_dtypes, date_dtypes, types_map)
    label_schema = create_spark_schema(label_types, string_dtypes, date_dtypes, types_map)

    # Set header to True or else it will be included as row
    train_psdf = spark.read.option("header", "true").csv(train_path, schema=train_schema)
    test_psdf = spark.read.option("header", "true").csv(test_path, schema=test_schema)
    label_psdf = spark.read.option("header", "true").csv(label_path, schema=label_schema)

    # Check schema
    print_splits(test_psdf.schema[:10])
    print(train_psdf.count())

    train_psdf.write.parquet("train_amex")
    test_psdf.write.parquet("test_amex")
    label_psdf.write.parquet("label_amex")


if __name__ == "__main__":
    main()
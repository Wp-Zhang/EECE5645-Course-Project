from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from ..utils import setup_logger, setup_timer

logger = setup_logger("Parallel Feature_Engineering")
timer = setup_timer(logger)

features_average = [
    "B_1",
    "B_2",
    "B_3",
    "B_4",
    "B_5",
    "B_6",
    "B_8",
    "B_9",
    "B_10",
    "B_11",
    "B_12",
    "B_13",
    "B_14",
    "B_15",
    "B_16",
    "B_17",
    "B_18",
    "B_19",
    "B_20",
    "B_21",
    "B_22",
    "B_23",
    "B_24",
    "B_25",
    "B_28",
    "B_29",
    "B_30",
    "B_32",
    "B_33",
    "B_37",
    "B_38",
    "B_39",
    "B_40",
    "B_41",
    "B_42",
    "D_39",
    "D_41",
    "D_42",
    "D_43",
    "D_44",
    "D_45",
    "D_46",
    "D_47",
    "D_48",
    "D_50",
    "D_51",
    "D_53",
    "D_54",
    "D_55",
    "D_58",
    "D_59",
    "D_60",
    "D_61",
    "D_62",
    "D_65",
    "D_66",
    "D_69",
    "D_70",
    "D_71",
    "D_72",
    "D_73",
    "D_74",
    "D_75",
    "D_76",
    "D_77",
    "D_78",
    "D_80",
    "D_82",
    "D_84",
    "D_86",
    "D_91",
    "D_92",
    "D_94",
    "D_96",
    "D_103",
    "D_104",
    "D_108",
    "D_112",
    "D_113",
    "D_114",
    "D_115",
    "D_117",
    "D_118",
    "D_119",
    "D_120",
    "D_121",
    "D_122",
    "D_123",
    "D_124",
    "D_125",
    "D_126",
    "D_128",
    "D_129",
    "D_131",
    "D_132",
    "D_133",
    "D_134",
    "D_135",
    "D_136",
    "D_140",
    "D_141",
    "D_142",
    "D_144",
    "D_145",
    "P_2",
    "P_3",
    "P_4",
    "R_1",
    "R_2",
    "R_3",
    "R_7",
    "R_8",
    "R_9",
    "R_10",
    "R_11",
    "R_14",
    "R_15",
    "R_16",
    "R_17",
    "R_20",
    "R_21",
    "R_22",
    "R_24",
    "R_26",
    "R_27",
    "S_3",
    "S_5",
    "S_6",
    "S_7",
    "S_9",
    "S_11",
    "S_12",
    "S_13",
    "S_15",
    "S_16",
    "S_18",
    "S_22",
    "S_23",
    "S_25",
    "S_26",
]
features_minimum = [
    "B_2",
    "B_4",
    "B_5",
    "B_9",
    "B_13",
    "B_14",
    "B_15",
    "B_16",
    "B_17",
    "B_19",
    "B_20",
    "B_28",
    "B_29",
    "B_33",
    "B_36",
    "B_42",
    "D_39",
    "D_41",
    "D_42",
    "D_45",
    "D_46",
    "D_48",
    "D_50",
    "D_51",
    "D_53",
    "D_55",
    "D_56",
    "D_58",
    "D_59",
    "D_60",
    "D_62",
    "D_70",
    "D_71",
    "D_74",
    "D_75",
    "D_78",
    "D_83",
    "D_102",
    "D_112",
    "D_113",
    "D_115",
    "D_118",
    "D_119",
    "D_121",
    "D_122",
    "D_128",
    "D_132",
    "D_140",
    "D_141",
    "D_144",
    "D_145",
    "P_2",
    "P_3",
    "R_1",
    "R_27",
    "S_3",
    "S_5",
    "S_7",
    "S_9",
    "S_11",
    "S_12",
    "S_23",
    "S_25",
]
features_maximum = [
    "B_1",
    "B_2",
    "B_3",
    "B_4",
    "B_5",
    "B_6",
    "B_7",
    "B_8",
    "B_9",
    "B_10",
    "B_12",
    "B_13",
    "B_14",
    "B_15",
    "B_16",
    "B_17",
    "B_18",
    "B_19",
    "B_21",
    "B_23",
    "B_24",
    "B_25",
    "B_29",
    "B_30",
    "B_33",
    "B_37",
    "B_38",
    "B_39",
    "B_40",
    "B_42",
    "D_39",
    "D_41",
    "D_42",
    "D_43",
    "D_44",
    "D_45",
    "D_46",
    "D_47",
    "D_48",
    "D_49",
    "D_50",
    "D_52",
    "D_55",
    "D_56",
    "D_58",
    "D_59",
    "D_60",
    "D_61",
    "D_63",
    "D_64",
    "D_65",
    "D_70",
    "D_71",
    "D_72",
    "D_73",
    "D_74",
    "D_76",
    "D_77",
    "D_78",
    "D_80",
    "D_82",
    "D_84",
    "D_91",
    "D_102",
    "D_105",
    "D_107",
    "D_110",
    "D_111",
    "D_112",
    "D_115",
    "D_116",
    "D_117",
    "D_118",
    "D_119",
    "D_121",
    "D_122",
    "D_123",
    "D_124",
    "D_125",
    "D_126",
    "D_128",
    "D_131",
    "D_132",
    "D_133",
    "D_134",
    "D_135",
    "D_136",
    "D_138",
    "D_140",
    "D_141",
    "D_142",
    "D_144",
    "D_145",
    "P_2",
    "P_3",
    "P_4",
    "R_1",
    "R_3",
    "R_5",
    "R_6",
    "R_7",
    "R_8",
    "R_10",
    "R_11",
    "R_14",
    "R_17",
    "R_20",
    "R_26",
    "R_27",
    "S_3",
    "S_5",
    "S_7",
    "S_8",
    "S_11",
    "S_12",
    "S_13",
    "S_15",
    "S_16",
    "S_22",
    "S_23",
    "S_24",
    "S_25",
    "S_26",
    "S_27",
]
features_last_occ = [
    "B_1",
    "B_2",
    "B_3",
    "B_4",
    "B_5",
    "B_6",
    "B_7",
    "B_8",
    "B_9",
    "B_10",
    "B_11",
    "B_12",
    "B_13",
    "B_14",
    "B_15",
    "B_16",
    "B_17",
    "B_18",
    "B_19",
    "B_20",
    "B_21",
    "B_22",
    "B_23",
    "B_24",
    "B_25",
    "B_26",
    "B_28",
    "B_29",
    "B_30",
    "B_32",
    "B_33",
    "B_36",
    "B_37",
    "B_38",
    "B_39",
    "B_40",
    "B_41",
    "B_42",
    "D_39",
    "D_41",
    "D_42",
    "D_43",
    "D_44",
    "D_45",
    "D_46",
    "D_47",
    "D_48",
    "D_49",
    "D_50",
    "D_51",
    "D_52",
    "D_53",
    "D_54",
    "D_55",
    "D_56",
    "D_58",
    "D_59",
    "D_60",
    "D_61",
    "D_62",
    "D_63",
    "D_64",
    "D_65",
    "D_69",
    "D_70",
    "D_71",
    "D_72",
    "D_73",
    "D_75",
    "D_76",
    "D_77",
    "D_78",
    "D_79",
    "D_80",
    "D_81",
    "D_82",
    "D_83",
    "D_86",
    "D_91",
    "D_96",
    "D_105",
    "D_106",
    "D_112",
    "D_114",
    "D_119",
    "D_120",
    "D_121",
    "D_122",
    "D_124",
    "D_125",
    "D_126",
    "D_127",
    "D_130",
    "D_131",
    "D_132",
    "D_133",
    "D_134",
    "D_138",
    "D_140",
    "D_141",
    "D_142",
    "D_145",
    "P_2",
    "P_3",
    "P_4",
    "R_1",
    "R_2",
    "R_3",
    "R_4",
    "R_5",
    "R_6",
    "R_7",
    "R_8",
    "R_9",
    "R_10",
    "R_11",
    "R_12",
    "R_13",
    "R_14",
    "R_15",
    "R_19",
    "R_20",
    "R_26",
    "R_27",
    "S_3",
    "S_5",
    "S_6",
    "S_7",
    "S_8",
    "S_9",
    "S_11",
    "S_12",
    "S_13",
    "S_16",
    "S_19",
    "S_20",
    "S_22",
    "S_23",
    "S_24",
    "S_25",
    "S_26",
    "S_27",
]


def aggregate_data_spark(df: DataFrame) -> DataFrame:
    """
    Adds new features to the input PySpark DataFrame: mean, minimum, maximum, and last value of certain features based on the customer id.

    This function takes a DataFrame with customer data and performs several aggregation operations.
    It computes the mean, minimum, maximum, and last value for a predefined set of features.

    Parameters:
    df (DataFrame): A PySpark DataFrame containing customer data.

    Returns:
    DataFrame: A new DataFrame containing the aggregated data.
    """

    # Defining the features for each aggregation
    features_avg = [f for f in features_average if f not in ["D_63", "D_64"]]
    features_min = [f for f in features_minimum if f not in ["D_63", "D_64"]]
    features_max = [f for f in features_maximum if f not in ["D_63", "D_64"]]
    features_last = [f for f in features_last_occ if f not in ["D_63", "D_64"]]

    # Grouping by customer_ID and calculating mean, min, max
    df_avg = df.groupBy("customer_ID").agg(
        *[F.avg(f).alias(f"{f}_avg") for f in features_avg]
    )
    df_min = df.groupBy("customer_ID").agg(
        *[F.min(f).alias(f"{f}_min") for f in features_min]
    )
    df_max = df.groupBy("customer_ID").agg(
        *[F.max(f).alias(f"{f}_max") for f in features_max]
    )
    df_first = df.groupBy("customer_ID").agg(
        *[F.first(f).alias(f"{f}_first") for f in features_last]
    )
    df_last = df.groupBy("customer_ID").agg(
        *[F.last(f).alias(f"{f}_last") for f in features_last]
    )

    # Renaming columns
    for feature in features_avg:
        df_avg = df_avg.withColumnRenamed(f"avg({feature})", f"{feature}_avg")
    for feature in features_min:
        df_min = df_min.withColumnRenamed(f"min({feature})", f"{feature}_min")
    for feature in features_max:
        df_max = df_max.withColumnRenamed(f"max({feature})", f"{feature}_max")

    # Calculating the last value for each feature
    # windowSpec = (
    #     Window.partitionBy("customer_ID")
    #     .orderBy("S_2")
    #     .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    # )
    # for feature in features_last:
    #     df_last = df.withColumn(
    #         f"{feature}_last", F.last(F.col(feature), True).over(windowSpec)
    #     )
    #     df_first = df.withColumn(
    #         f"{feature}_first", F.first(F.col(feature), True).over(windowSpec)
    #     )

    # # Selecting only the last and first row for each customer
    # df_last = df_last.select(
    #     ["customer_ID"] + [f"{feature}_last" for feature in features_last]
    # ).distinct()
    # df_first = df_last.select(
    #     ["customer_ID"] + [f"{feature}_first" for feature in features_last]
    # ).distinct()

    # Joining the DataFrames
    df_combined = df_avg.join(df_min, "customer_ID", "outer")
    df_combined = df_combined.join(df_max, "customer_ID", "outer")
    df_combined = df_combined.join(df_last, "customer_ID", "outer")
    df_combined = df_combined.join(df_first, "customer_ID", "outer")

    return df_combined


def after_pay_features_spark(df: DataFrame) -> DataFrame:
    """
    Adds new "after-pay" features to the input PySpark DataFrame.
    """

    for bcol in (
        [f"B_{i}" for i in [1, 2, 3, 4, 5, 9, 11, 14, 17, 24]]
        + ["D_39", "D_131"]
        + [f"S_{i}" for i in [16, 23]]
    ):
        for pcol in ["P_2", "P_3"]:
            if bcol in df.columns:
                df = df.withColumn(f"{bcol}-{pcol}", F.col(bcol) - F.col(pcol))
                features_average.append(f"{bcol}-{pcol}")

    return df


def last_first_difference_features_spark(df: DataFrame) -> DataFrame:
    for col in df.columns:
        if col.endswith("first"):
            base_feature = col[:-6]  # Removes '_first' suffix
            last_col = base_feature + "_last"

            if last_col in df.columns:
                df = df.withColumn(
                    f"{base_feature}_last_first_diff", F.col(last_col) - F.col(col)
                )

    return df


@timer
def feature_engineer_spark(df: DataFrame) -> DataFrame:
    df = after_pay_features_spark(df)
    df = aggregate_data_spark(df)
    df = last_first_difference_features_spark(df)

    logger.info("parallel feature engineering done")

    return df

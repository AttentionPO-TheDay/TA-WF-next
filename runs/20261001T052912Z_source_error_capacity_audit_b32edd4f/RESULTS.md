# 源期错误与数据量审计

可用训练样本18553，比150条/类增加3253；每类min/median/max=156/184.0/190。保持类均衡最多每类156条，不可承诺扩到300/500条。

两个模型全部六seed都错的验证样本：54/510。仅描述已观察valid错误，不是独立确认。

{
  "source_total": 19439,
  "eligible_total": 18553,
  "additional_over150": 3253,
  "per_class_min": 156,
  "per_class_median": 184.0,
  "per_class_max": 190,
  "balanced_max_per_class": 156,
  "excluded": {
    "overlap": 168,
    "conflict": 0,
    "structure": 0,
    "duplicate": 718
  },
  "source_stored_width": 5000,
  "selected_valid_longer_than5000": 23,
  "selected_valid_at5000": 23,
  "both_models_all_seeds_wrong": 54,
  "models": {
    "mlp": {
      "mean_accuracy": 0.7215686274509804,
      "all_seed_wrong": 83,
      "any_seed_wrong": 204,
      "all_seed_correct": 306,
      "pairwise_error_jaccard": [
        0.5652173913043478,
        0.5271739130434783,
        0.5810055865921788
      ],
      "wrong_confidence_ge_09": 86,
      "wrong_predictions": 426,
      "length_bins": [
        {
          "length_range": [
            1,
            1000
          ],
          "samples": 246,
          "mean_error_rate": 0.3252032520325203,
          "all_seed_wrong": 51
        },
        {
          "length_range": [
            1001,
            2500
          ],
          "samples": 189,
          "mean_error_rate": 0.25396825396825395,
          "all_seed_wrong": 27
        },
        {
          "length_range": [
            2501,
            4999
          ],
          "samples": 52,
          "mean_error_rate": 0.17307692307692307,
          "all_seed_wrong": 3
        },
        {
          "length_range": [
            5000,
            5000
          ],
          "samples": 23,
          "mean_error_rate": 0.21739130434782608,
          "all_seed_wrong": 2
        }
      ],
      "top20_confusions": [
        {
          "count_across_seeds": 8,
          "true_class": 47,
          "predicted_class": 101
        },
        {
          "count_across_seeds": 6,
          "true_class": 101,
          "predicted_class": 47
        },
        {
          "count_across_seeds": 5,
          "true_class": 95,
          "predicted_class": 50
        },
        {
          "count_across_seeds": 4,
          "true_class": 100,
          "predicted_class": 16
        },
        {
          "count_across_seeds": 4,
          "true_class": 42,
          "predicted_class": 72
        },
        {
          "count_across_seeds": 4,
          "true_class": 23,
          "predicted_class": 73
        },
        {
          "count_across_seeds": 4,
          "true_class": 13,
          "predicted_class": 95
        },
        {
          "count_across_seeds": 4,
          "true_class": 0,
          "predicted_class": 30
        },
        {
          "count_across_seeds": 3,
          "true_class": 101,
          "predicted_class": 61
        },
        {
          "count_across_seeds": 3,
          "true_class": 96,
          "predicted_class": 19
        },
        {
          "count_across_seeds": 3,
          "true_class": 92,
          "predicted_class": 70
        },
        {
          "count_across_seeds": 3,
          "true_class": 90,
          "predicted_class": 22
        },
        {
          "count_across_seeds": 3,
          "true_class": 87,
          "predicted_class": 58
        },
        {
          "count_across_seeds": 3,
          "true_class": 77,
          "predicted_class": 76
        },
        {
          "count_across_seeds": 3,
          "true_class": 77,
          "predicted_class": 70
        },
        {
          "count_across_seeds": 3,
          "true_class": 74,
          "predicted_class": 92
        },
        {
          "count_across_seeds": 3,
          "true_class": 68,
          "predicted_class": 101
        },
        {
          "count_across_seeds": 3,
          "true_class": 63,
          "predicted_class": 7
        },
        {
          "count_across_seeds": 3,
          "true_class": 60,
          "predicted_class": 33
        },
        {
          "count_across_seeds": 3,
          "true_class": 59,
          "predicted_class": 3
        }
      ],
      "top20_confusion_error_fraction": 0.176056338028169,
      "majority_vote_accuracy": 0.7431372549019608,
      "all_wrong_rows": [
        210,
        1578,
        880,
        824,
        722,
        1537,
        1497,
        1155,
        2006,
        225,
        2012,
        813,
        1961,
        2,
        473,
        330,
        1782,
        1711,
        389,
        1920,
        856,
        907,
        720,
        1430,
        143,
        1116,
        1992,
        2044,
        1482,
        1416,
        1662,
        1227,
        102,
        809,
        236,
        255,
        1057,
        566,
        161,
        1692,
        983,
        1304,
        95,
        1151,
        961,
        1229,
        1528,
        1985,
        777,
        17,
        452,
        1049,
        670,
        265,
        2136,
        1941,
        1273,
        873,
        1795,
        1807,
        1370,
        278,
        1775,
        557,
        2052,
        1316,
        1203,
        1240,
        305,
        1309,
        1608,
        1168,
        779,
        1058,
        1955,
        572,
        976,
        1079,
        158,
        1757,
        2126,
        343,
        1087
      ]
    },
    "transformer": {
      "mean_accuracy": 0.7156862745098039,
      "all_seed_wrong": 75,
      "any_seed_wrong": 222,
      "all_seed_correct": 288,
      "pairwise_error_jaccard": [
        0.494949494949495,
        0.4467005076142132,
        0.5454545454545454
      ],
      "wrong_confidence_ge_09": 190,
      "wrong_predictions": 435,
      "length_bins": [
        {
          "length_range": [
            1,
            1000
          ],
          "samples": 246,
          "mean_error_rate": 0.3252032520325203,
          "all_seed_wrong": 45
        },
        {
          "length_range": [
            1001,
            2500
          ],
          "samples": 189,
          "mean_error_rate": 0.2698412698412698,
          "all_seed_wrong": 24
        },
        {
          "length_range": [
            2501,
            4999
          ],
          "samples": 52,
          "mean_error_rate": 0.1794871794871795,
          "all_seed_wrong": 5
        },
        {
          "length_range": [
            5000,
            5000
          ],
          "samples": 23,
          "mean_error_rate": 0.2028985507246377,
          "all_seed_wrong": 1
        }
      ],
      "top20_confusions": [
        {
          "count_across_seeds": 6,
          "true_class": 101,
          "predicted_class": 47
        },
        {
          "count_across_seeds": 6,
          "true_class": 95,
          "predicted_class": 50
        },
        {
          "count_across_seeds": 6,
          "true_class": 47,
          "predicted_class": 101
        },
        {
          "count_across_seeds": 6,
          "true_class": 13,
          "predicted_class": 95
        },
        {
          "count_across_seeds": 5,
          "true_class": 87,
          "predicted_class": 58
        },
        {
          "count_across_seeds": 5,
          "true_class": 58,
          "predicted_class": 68
        },
        {
          "count_across_seeds": 5,
          "true_class": 50,
          "predicted_class": 95
        },
        {
          "count_across_seeds": 4,
          "true_class": 43,
          "predicted_class": 8
        },
        {
          "count_across_seeds": 4,
          "true_class": 4,
          "predicted_class": 8
        },
        {
          "count_across_seeds": 3,
          "true_class": 95,
          "predicted_class": 13
        },
        {
          "count_across_seeds": 3,
          "true_class": 82,
          "predicted_class": 23
        },
        {
          "count_across_seeds": 3,
          "true_class": 70,
          "predicted_class": 12
        },
        {
          "count_across_seeds": 3,
          "true_class": 66,
          "predicted_class": 97
        },
        {
          "count_across_seeds": 3,
          "true_class": 60,
          "predicted_class": 42
        },
        {
          "count_across_seeds": 3,
          "true_class": 59,
          "predicted_class": 3
        },
        {
          "count_across_seeds": 3,
          "true_class": 48,
          "predicted_class": 42
        },
        {
          "count_across_seeds": 3,
          "true_class": 45,
          "predicted_class": 29
        },
        {
          "count_across_seeds": 3,
          "true_class": 37,
          "predicted_class": 86
        },
        {
          "count_across_seeds": 3,
          "true_class": 32,
          "predicted_class": 8
        },
        {
          "count_across_seeds": 3,
          "true_class": 25,
          "predicted_class": 18
        }
      ],
      "top20_confusion_error_fraction": 0.1839080459770115,
      "majority_vote_accuracy": 0.7509803921568627,
      "all_wrong_rows": [
        210,
        1146,
        1578,
        880,
        8,
        824,
        144,
        1497,
        141,
        1934,
        1155,
        2006,
        225,
        813,
        1961,
        473,
        1711,
        1920,
        856,
        720,
        1430,
        143,
        1356,
        1832,
        152,
        793,
        1857,
        1416,
        1662,
        1227,
        102,
        809,
        236,
        1057,
        566,
        1692,
        1387,
        983,
        1304,
        95,
        1151,
        961,
        1229,
        1528,
        777,
        17,
        670,
        265,
        1395,
        2136,
        1821,
        1941,
        2120,
        1273,
        201,
        873,
        631,
        1697,
        428,
        1807,
        1775,
        2052,
        367,
        1432,
        1240,
        1309,
        1608,
        779,
        572,
        976,
        1419,
        158,
        1757,
        2126,
        1087
      ]
    }
  },
  "elapsed_seconds": 13.79870537854731,
  "future_access": false
}

已核验6份checkpoint在CPU上重算与历史GPU预测逐条一致。保存置信度仅用于本轮诊断，不用于适应或梯度。source/valid完全方向重复已排除，source150嵌套保留；允许数据用途未扩展至未来。

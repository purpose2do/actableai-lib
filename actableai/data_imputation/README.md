# Automatic Data Imputation

To use the fixer you can do:
```Python
from data import DataFrame

BROKEN_DATA_PATH = "broken.csv"
df = DataFrame(BROKEN_DATA_PATH)
fixed_df = df.auto_fix()
```

## Manually set Column Type
The auto detector might not very accurate sometimes, you can manually override the detected type to boost the fix accuracy. E.g.
```python
from meta import ColumnType

df.override_column_type("Score", ColumnType.Percentage)
```

## Error Detectors
There are three types of error detector:
1. NullDetector: detect the nan values
2. MisplacedDetector: use regex to track if any value in the column is not as expected
3. ValidationDetector: use error constrains to describe the accepted values, using the relationship between two or more columns.


## MisplacedDetector
The `MisplacedDetector` is able to let user set the expected value/regex they want for a column. User can choose to use regex or value match.

Parameters for `MatchRule`:
- column: column name.
- match_str: can be a value or a regex. N.B. the regex will be surrounding by ^ and $ in order to do a full string match.
- is_regex: mark `match_str` as a value or a regex.
- match_as_mismatch: when True set the cell matched with `match_str` as misplaced, and vice versa.

There are two types of rules: `MatchStrRule` and `MatchNumRule`.

The following example shows that the user mark all value equals to `Bachelors` in `education` column is a fault value.

```python
from data import DataFrame
from error_detector import MisplacedDetector, MatchRules, MatchStrRule, ConditionOp

df = DataFrame(data={"education": []})

detectors = [MisplacedDetector(customize_rules=MatchRules(
    preset_rules=[PresetRuleName.SmartTemperature],
    customize_rules=[
        MatchStrRule(
            column="education",
            match_str="Bachelors", 
            is_regex=False, 
            op=ConditionOp.EQ)
    ]
))]
errors = df.detect_error(*detectors)
```

The following example shows that the user mark all value larger than to 20 in `age` column is a fault value.
```python
from data import DataFrame
from error_detector import MisplacedDetector, MatchRules, MatchNumRule, ConditionOp

df = DataFrame(data={"age": []})

detectors = [MisplacedDetector(customize_rules=MatchRules(
    preset_rules=[],
    customize_rules=[
        MatchNumRule(
            column="age",
            match_val=20, 
            op=ConditionOp.LT)
    ]
))]
errors = df.detect_error(*detectors)
```

There are also two preset rules: `PresetRuleName.SmartTemperature`, `PresetRuleName.SmartTemperature`. They are just a pre-write MatchStrRule using regex and will detect the cell as fault when the value does not match the format.

```python
from data import DataFrame
from error_detector import MisplacedDetector, MatchRules, PresetRuleName

df = DataFrame(data={"a": []})

detectors = [MisplacedDetector(customize_rules=MatchRules(
    preset_rules=[PresetRuleName.SmartTemperature],
    customize_rules=[]
))]
```



## Error Constrains

It also supports the constraint to describe the error in the data.
E.g.
```Python
from error_detector import ValidationDetector

ValidationDetector.from_constraints(<constraints>)
```

The syntax for the constraints:
```
a>b -> c<b
```
This means _WHEN_ `a>b` _THEN_ `c<b` is an error, where `a`,`b`,`c`,`d` are column names. The error scan is doing row by row.

This detector is also support multiple conditions, you can connect different conditions with `OR` or a new line.

## Environment Variable

**SMALL_DATASET_LENGTH_THRESHOLD**: This project is using a hardcoded value to distinguish if the dataset is small or large. The default value right now is 300 lines.

**MATCH_ROW_NUM_THRESHOLD**: We are currently using a few hardcoded rules to determine the columns types, this variable is the percentage of matched rows.

**STRING_LENGTH_THRESHOLD**: The threshold for the length of text to determine whether it's a long text.

**CATEGORY_PERCENT_THRESHOLD**: The number of distinct value percentage in the column to determine if it's a category column.

**TYPO_APPEARANCE_FREQUENCY_THRESHOLD**: The value (0 to 1) represent the appearance percentage in the rows which other columns have same values as the other rows.

## What's Next

1. ~~Support the column that the value is a combination of number and tag.~~
2. Use pretrained model to detect the column type.
3. Use pretrained model for data embedding.
4. Data segmentation to solve unbiased data with small data number in specific cluster.
5. ~~Extend error detector with a misplaced value detector. E.g. in percentage column, we are not accepting xx% as a valid value.~~
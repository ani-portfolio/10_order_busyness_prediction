import logging

logger = logging.getLogger(__name__)


def validate_schema(df, expected_schema):
    """

    """

    missing_cols = set(expected_schema.keys()) - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    for col, expected_type in expected_schema.items():
        if df[col].dtype != expected_type:
            raise TypeError(f"Column {col} has type {df[col].dtype}, expected {expected_type}")
    
    logger.info("Schema validation passed")


def validate_nulls(df, null_rules):
    """
    
    """

    for col, rule in null_rules.items():
        null_count = df[col].isna().sum()
        null_pct = null_count / len(df)
        
        if rule == 'no_nulls' and null_count > 0:
            raise ValueError(f"Column {col} has {null_count} nulls, expected none")
        
        if isinstance(rule, dict) and 'max_pct' in rule:
            if null_pct > rule['max_pct']:
                raise ValueError(f"Column {col} has {null_pct} nulls, exceeds {rule['max_pct']}")
    
    logger.info("Null validation passed")

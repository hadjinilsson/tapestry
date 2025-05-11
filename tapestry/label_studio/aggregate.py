import pandas as pd
from itertools import combinations


def aggregate_labels(ls_data, min_group_size=100):
    """
    Aggregates detailed labels from Label Studio annotations based on
    combinations of multi-choice fields (e.g., direction, orientation).

    Parameters:
        ls_data: List of Label Studio task dictionaries (from JSON export)
        min_group_size: Minimum number of samples required to preserve a unique combination

    Returns:
        Modified ls_data with 'label_detailed' written into rectanglelabels field
    """

    # Step 1: Extract label and choice combinations
    ann_lacs = []

    for task in ls_data:
        result = task.get("annotations", [])[0].get("result", [])
        for item in result:
            if item["type"] not in item["value"]:
                continue
            region_id = item.get("id") or f"rect_{item['value'].get('x', 0)}_{item['value'].get('y', 0)}"
            item_value = set(item["value"][item["type"]])
            value_str = "_".join(sorted(item_value))
            ann_lacs.append({
                'region_id': region_id,
                'type': item['from_name'],
                'value': value_str,
            })

    # Step 2: Pivot to wide format: one column per choice field
    ann_df = pd.DataFrame(ann_lacs).pivot(columns='type', index='region_id')
    ann_df.columns = ann_df.columns.droplevel(0)
    ann_df['label_detailed'] = ann_df['label'].apply(lambda x: " ".join(sorted({x})))
    ann_df['assigned'] = False

    # Step 3: Aggregate combinations per base label
    for base_label in ann_df['label'].unique():
        grouped_choices = list(
            ann_df[ann_df['label'] == base_label]
            .dropna(axis=1)
            .drop(columns=['label', 'label_detailed', 'assigned'])
            .columns
        )

        while True:
            group = ann_df[(ann_df['label'] == base_label) & (~ann_df['assigned'])]
            grouped = group.groupby(grouped_choices, as_index=False).size()
            large = grouped[grouped['size'] >= min_group_size]
            small = grouped[grouped['size'] < min_group_size]

            # Assign detailed labels to large groups
            for _, row in large.iterrows():
                query = f"label == '{base_label}' & assigned == False"
                choice_vals = []
                for choice in grouped_choices:
                    query += f" & {choice} == '{row[choice]}'"
                    choice_vals.append(row[choice])
                detailed = " ".join([base_label] + sorted(choice_vals)).capitalize()
                matched_ids = ann_df.query(query).index
                ann_df.loc[matched_ids, 'label_detailed'] = detailed
                ann_df.loc[matched_ids, 'assigned'] = True

            if len(small) == 0 or len(grouped_choices) == 1:
                break  # Done or can't simplify further

            # Try dropping one choice dimension to merge remaining small groups
            metrics = []
            for choice_set in combinations(grouped_choices, len(grouped_choices) - 1):
                collapsed = small.groupby(list(choice_set), as_index=False)['size'].sum()
                collapsed['gap'] = (min_group_size - collapsed['size']).clip(lower=0)
                metrics.append({
                    'choices': list(choice_set),
                    'min_gap': collapsed['gap'].min(),
                    'sum_gap': collapsed['gap'].sum(),
                    'num_classes': -len(collapsed),
                    'std': collapsed['gap'].std(),
                })

            # Choose best candidate combination to merge by gap/complexity
            metrics_df = pd.DataFrame(metrics)
            grouped_choices = metrics_df.sort_values(['min_gap', 'sum_gap', 'num_classes', 'std']).iloc[0]['choices']

    # Step 4: Write updated label_detailed back into LS data
    for task in ls_data:
        result = task.get("annotations", [])[0].get("result", [])
        for item in result:
            if item["type"] != "rectanglelabels":
                continue
            region_id = item.get("id") or f"rect_{item['value'].get('x', 0)}_{item['value'].get('y', 0)}"
            detailed = ann_df.loc[region_id, 'label_detailed']
            item["value"]["rectanglelabels"] = [detailed]

    return ls_data
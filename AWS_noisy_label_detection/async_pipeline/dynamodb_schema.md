# DynamoDB Table Schema

Table name: `noise_detection_results`

## Primary Key
- `job_id` (Partition Key)
- `sample_id` (Sort Key)

## Example Attributes
- `observed_label`
- `observed_label_name`
- `predicted_label`
- `predicted_label_name`
- `noise_score`
- `prob_observed_label`
- `created_at`
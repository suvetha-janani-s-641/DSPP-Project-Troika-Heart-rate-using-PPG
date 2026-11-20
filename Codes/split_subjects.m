function [train_data, test_data] = split_subjects(full_dataset, subject_ids, target_subject)
% Isolates a specific subject's data rows for Leave-One-Subject-Out validation
test_mask = (subject_ids == target_subject);
test_data = full_dataset(test_mask, :, :);
train_data = full_dataset(~test_mask, :, :);
end
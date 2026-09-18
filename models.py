import torch
import torch.nn as nn

# dictionary to hold the different condition
# comparions to be fed to the model

# keys for condition names
conditions = [
    "text",
    "image",
    "early",
    "intermediate",
    "late",
    "learned"
]

hidden_width_grid = [128, 256, 512]
learning_rate_grid = [1e-3, 3e-4]

def build_config(condition, hidden_width, learning_rate):

    if condition in conditions:
        return {
            "condition": condition,
            "hidden_width": hidden_width,
            "learning_rate": learning_rate,
            "dropout": 0.2,
            "input_dim": 768,
            "n_classes": 3,
            "seed": 7
        }

test_config_x = build_config("text", 256, 1e-3)
test_config_y = build_config("image", 128, 3e-4)

test_config_x["hidden_width"] = 123

print(test_config_x is test_config_y)
print(test_config_y["hidden_width"])


def build_block(input_dim, output_dim, dropout):
    # one projection block
    # linear layer, non-linearity, dropout

    block = nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU(),
        nn.Dropout(dropout)
    )

    return block

block = build_block(768, 256, 0.2)
print(block)


def build_head(input_dim, hidden_width, dropout, n_classes):
    # classifier head
    # input_dim varies by condition
    head = nn.Sequential(
        build_block(input_dim, hidden_width, dropout),
        nn.Linear(hidden_width, n_classes)
    )
    return head


head = build_head(512, 256, 0.2, 3)
print(head)


# condition model
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

# grid with fixed settings
# setting this up means that every condition has a limited amount of settings to try
hidden_width_grid = [128, 256, 512]
learning_rate_grid = [1e-3, 3e-4]

# makes a settings sheet for each run
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


# take a set of numbers and shrink them to a smaller learned summary
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


# classifier
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

# example from PyTorch docs
# class Model(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5)
#         self.conv2 = nn.Conv2d(20, 20, 5)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         return F.relu(self.conv2(x))


class ConditionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.condition = config["condition"]

        input_dim = config["input_dim"]
        hidden_width = config["hidden_width"]
        dropout = config["dropout"]
        n_classes = config["n_classes"]

        

        # text only condition
        if self.condition == "text":
            self.text_tower = build_block(input_dim, hidden_width, dropout)
            self.head = build_head(hidden_width, hidden_width, dropout, n_classes)

        # image only condition
        elif self.condition == "image":
            self.image_tower = build_block(input_dim, hidden_width, dropout)
            self.head = build_head(hidden_width, hidden_width, dropout, n_classes)  

        # early fusion
        elif self.condition == "early":
            fused_width = hidden_width
            self.early_tower = build_block(2 * input_dim, hidden_width, dropout)
            self.head = build_head(fused_width, hidden_width, dropout, n_classes)

        # intermediate fusion
        elif self.condition == "intermediate":
            fused_width = 2 * hidden_width
            self.text_tower = build_block(input_dim, hidden_width, dropout)
            self.image_tower = build_block(input_dim, hidden_width, dropout)
            self.head = build_head(fused_width, hidden_width, dropout, n_classes)


        # late fusion
        # learned fusion  


    def forward(self, text_embedding, image_embedding):
        # text only
        if self.condition == "text":
            hidden = self.text_tower(text_embedding)

        # image only
        elif self.condition == "image":
            hidden = self.image_tower(image_embedding)

        # early fusion
        elif self.condition == "early":
            combined = torch.cat([text_embedding, image_embedding], dim=1)
            hidden = self.early_tower(combined)

        # intermediate fusion
        elif self.condition == "intermediate":
            text_hidden = self.text_tower(text_embedding)
            image_hidden = self.image_tower(image_embedding)
            hidden = torch.cat([text_hidden, image_hidden], dim=1)
            
        # late fusion
        # learned fusion  

        return self.head(hidden)



text_model = ConditionModel(build_config("text", 256, 1e-3))
image_model = ConditionModel(build_config("image", 256, 1e-3))

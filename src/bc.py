import torch


def train_bc(model, obs, actions, steps, batch_size, lr):
    obs = torch.as_tensor(obs)
    actions = torch.as_tensor(actions)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        idx = torch.randint(len(obs), (batch_size,))
        loss = ((model(obs[idx]) - actions[idx]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

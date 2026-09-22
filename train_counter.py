import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from stack_attention import DataStructureTransLayer, data_structure

# data structure definition

counter = data_structure(
    init = lambda: torch.zeros(1),
    actions = {
        'noop': lambda state: state,
        'increment': lambda state, item: state + item
    },
    readout = lambda state: state,
    readout_transform = nn.Sequential(
        nn.Linear(1, 32),
        nn.SiLU(),
        nn.Linear(32, 16)
    ),
    render = lambda state: f'count={state.item():.1f}'
)

# models

class CounterModel(nn.Module):
    def __init__(self, dim = 32):
        super().__init__()
        self.embed = nn.Embedding(2, dim)
        self.layer = DataStructureTransLayer(
            dim = dim,
            num_heads = 1,
            data_structure = counter,
            add_residual = False
        )
        self.to_pred = nn.Linear(dim, 1)

    def introspect(self, x, **kwargs):
        tokens = self.embed(x)
        return self.layer.introspect(tokens, **kwargs)

    def forward(self, x, **kwargs):
        tokens = self.embed(x)
        out, state = self.layer(tokens, recurrent = True, **kwargs)
        return self.to_pred(out).squeeze(-1), state

class RNNBaseline(nn.Module):
    def __init__(self, dim = 32):
        super().__init__()
        self.embed = nn.Embedding(2, dim)
        self.cell = nn.Sequential(
            nn.Linear(dim + dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.to_pred = nn.Linear(dim, 1)

    def forward(self, x):
        tokens = self.embed(x)
        b, n, d = tokens.shape
        h = torch.zeros(b, d, device = x.device)
        preds = []
        for t in range(n):
            h = self.cell(torch.cat((tokens[:, t], h), dim = -1)) + h
            preds.append(self.to_pred(h).squeeze(-1))
        return torch.stack(preds, dim = 1)

# main training and evaluation

def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dim = 32
    counter_model = CounterModel(dim = dim).to(device)
    rnn_model = RNNBaseline(dim = dim).to(device)

    opt_counter = torch.optim.Adam(counter_model.parameters(), lr = 1e-2)
    opt_rnn = torch.optim.Adam(rnn_model.parameters(), lr = 1e-2)

    train_steps = 800
    batch_size = 32
    min_train_len, max_train_len = 4, 12

    counter_losses = []
    rnn_losses = []

    sched_counter = torch.optim.lr_scheduler.CosineAnnealingLR(opt_counter, T_max = train_steps, eta_min = 1e-4)
    sched_rnn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_rnn, T_max = train_steps, eta_min = 1e-4)

    print(f"Training on sequence lengths {min_train_len} to {max_train_len} for {train_steps} steps...")

    for step in range(train_steps):
        seq_len = torch.randint(min_train_len, max_train_len + 1, (1,)).item()
        x = torch.randint(0, 2, (batch_size, seq_len), device = device)
        target = x.float().cumsum(dim = -1)

        # anneal the action temperature so the counter commits to discrete actions

        temperature = 1. - 0.9 * step / train_steps

        # counter update

        pred_c, _ = counter_model(x, action_temperature = temperature)
        loss_c = F.mse_loss(pred_c, target)
        opt_counter.zero_grad()
        loss_c.backward()
        opt_counter.step()
        sched_counter.step()

        # rnn update

        pred_r = rnn_model(x)
        loss_r = F.mse_loss(pred_r, target)
        opt_rnn.zero_grad()
        loss_r.backward()
        opt_rnn.step()
        sched_rnn.step()

        counter_losses.append(loss_c.item())
        rnn_losses.append(loss_r.item())

        if (step + 1) % 200 == 0:
            print(f'Step {step + 1:>3}/{train_steps} | Counter MSE: {loss_c.item():.4f} | RNN MSE: {loss_r.item():.4f}')

    # evaluation across lengths (including unseen out-of-distribution lengths)

    counter_model.eval()
    rnn_model.eval()

    eval_lengths = list(range(4, 65, 4))
    counter_eval_mses = []
    rnn_eval_mses = []
    eval_batch_size = 128

    print('\nEvaluating length generalization (lengths 4 to 64)...')

    with torch.no_grad():
        for l in eval_lengths:
            x_test = torch.randint(0, 2, (eval_batch_size, l), device = device)
            y_test = x_test.float().sum(dim = -1)

            pred_c, _ = counter_model(x_test, hard_action = True)
            pred_r = rnn_model(x_test)

            mse_c = F.mse_loss(pred_c[:, -1], y_test).item()
            mse_r = F.mse_loss(pred_r[:, -1], y_test).item()

            counter_eval_mses.append(mse_c)
            rnn_eval_mses.append(mse_r)

            tag = '(Train Range)' if l <= max_train_len else '(OOD Extrapolation)'
            print(f'Length {l:>2} {tag:<20} | Counter MSE: {mse_c:>8.4f} | RNN MSE: {mse_r:>8.4f}')

    # introspection sample on a test sequence

    sample_seq = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1]], device = device)
    traj = counter_model.introspect(sample_seq)

    print('\n=== Counter Introspection Trajectory on Sample Sequence ===')
    print(f'Tokens: {sample_seq[0].tolist()} (Ground truth sum = {sample_seq.sum().item()})\n')
    print(traj.summary(batch_idx = 0, head_idx = 0))

    # plotting charts

    import seaborn as sns

    sns.set_theme(style = 'whitegrid')
    fig, (ax_train, ax_gen, ax_actions) = plt.subplots(1, 3, figsize = (18, 5))

    # panel 1: training loss

    sns.lineplot(data = counter_losses, ax = ax_train, label = 'Counter (DataStructureTransLayer)', color = '#10b981', alpha = 0.8)
    sns.lineplot(data = rnn_losses, ax = ax_train, label = 'RNN Baseline', color = '#ef4444', alpha = 0.8)
    ax_train.set(title = 'Training Loss (Sequence Lengths 4-12)', xlabel = 'Optimization Step', ylabel = 'MSE Loss', yscale = 'log')
    ax_train.legend()

    # panel 2: length generalization

    sns.lineplot(x = eval_lengths, y = counter_eval_mses, ax = ax_gen, marker = 'o', label = 'Counter', color = '#10b981', linewidth = 2)
    sns.lineplot(x = eval_lengths, y = rnn_eval_mses, ax = ax_gen, marker = 's', label = 'RNN Baseline', color = '#ef4444', linewidth = 2, linestyle = '--')
    ax_gen.axvspan(min_train_len, max_train_len, color = 'gray', alpha = 0.15, label = 'Training Length Range')
    ax_gen.set(title = 'Length Generalization MSE vs Sequence Length', xlabel = 'Sequence Length', ylabel = 'Test MSE Loss', yscale = 'log')
    ax_gen.legend()

    # panel 3: action probabilities on test sequence

    actions_mat = traj.action_matrix(batch_idx = 0, head_idx = 0).detach().cpu().numpy()
    seq_len = sample_seq.shape[1]
    tokens_labels = [f'tok={sample_seq[0, t].item()}' for t in range(seq_len)]

    ax_actions.bar(range(seq_len), actions_mat[:, 1], label = 'P(increment)', color = '#3b82f6', alpha = 0.8)
    ax_actions.bar(range(seq_len), actions_mat[:, 0], bottom = actions_mat[:, 1], label = 'P(noop)', color = '#9ca3af', alpha = 0.6)
    ax_actions.set_xticks(range(seq_len), labels = tokens_labels, rotation = 45, ha = 'right')
    ax_actions.set(title = 'Introspection: Action Probabilities per Token', xlabel = 'Time Step (Token Value)', ylabel = 'Action Probability', ylim = (0, 1.05))
    ax_actions.legend()

    sns.despine(fig = fig, top = True, right = True)
    plt.tight_layout()

    chart_path = 'counter_generalization.png'
    plt.savefig(chart_path, dpi = 150)
    plt.close()

    print(f'\nCharts successfully saved to {chart_path}')

if __name__ == '__main__':
    main()

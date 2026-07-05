"""Grid search over knowledge-distillation alpha/temperature combinations.

Trains one fresh student per (alpha, temperature) combination, each logged to
its own TensorBoard run so the results can be compared side by side. Run this
once and let it finish unattended.
"""

from torch.utils.tensorboard import SummaryWriter

from main import (
    NET_CLASS as STUDENT_CLASS,
    device,
    get_data_loaders,
    load_teacher,
    save_model,
    test,
    train_and_validate_distillation,
)

ALPHA_GRID = [0.1, 0.2, 0.3]
TEMPERATURE_GRID = [2.0, 4.0, 6.0]


def run_grid_search():
    """Train one student per (alpha, temperature) combination and record test accuracy."""
    train_dataloader, val_dataloader, test_dataloader = get_data_loaders()
    teacher = load_teacher()

    for alpha in ALPHA_GRID:
        for temperature in TEMPERATURE_GRID:
            run_name = f"{STUDENT_CLASS.__name__}_distilled_a{alpha}_T{temperature}"
            print(f"\n=== Training {run_name} ===")

            student = STUDENT_CLASS().to(device)
            writer = SummaryWriter(f"runs/{run_name}")

            train_and_validate_distillation(
                train_dataloader, val_dataloader, teacher, student, writer,
                temperature=temperature, alpha=alpha,
            )
            save_model(student, run_name)
            test_accuracy = test(test_dataloader, student)
            writer.add_scalar("Accuracy/test", test_accuracy)
            writer.close()

if __name__ == "__main__":
    run_grid_search()

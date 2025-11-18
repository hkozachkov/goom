"""Console entry point for goom."""

from .custom_exp import demo_gradients


def main() -> None:
    print("Input    | Custom grad")
    print("---------|-------------")
    for x_val, grad in demo_gradients():
        print(f"{x_val:>7.1f} | {grad:>11.6f}")


if __name__ == "__main__":
    main()


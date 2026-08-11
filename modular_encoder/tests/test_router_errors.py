from router.input_router import InputRouter

def main() -> None:
    router = InputRouter()
    tests = [
        lambda: router.route(),
        lambda: router.route(file_paths=["data/not_found.jpg"]),
        lambda: router.route(file_paths="data/test_image.jpg"),
    ]
    for test in tests:
        try:
            test()
            raise AssertionError("Expected an exception.")
        except (ValueError, FileNotFoundError, TypeError):
            pass
    print("Router error tests passed.")

if __name__ == "__main__":
    main()

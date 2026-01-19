def display_results(df, stats):
    print("\n=== Mixture CN Prediction ===\n")
    print(df.head())

    if stats:
        print("\nMetrics:")
        for k, v in stats.items():
            print(f"{k}: {v:.3f}")

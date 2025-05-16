    import pandas as pd
    import sqlite3
    import great_expectations as ge
    from great_expectations.core.batch import RuntimeBatchRequest

    def clean_reddit_titles_from_db(db_path="../db/social_media.db", table_name="reddit_posts"):
        conn = None
        try:
            # Connect to SQLite database and load table
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            original_count = len(df)

            # Create an in-memory (ephemeral) GE Data Context
            context = ge.get_context()

            # Create a temporary Expectation Suite directly
            suite_name = "temp_suite"
            context.add_or_update_expectation_suite(expectation_suite_name=suite_name)

            # Create batch request
            batch_request = RuntimeBatchRequest(
                datasource_name="my_pandas_datasource",
                data_connector_name="default_runtime_data_connector_name",
                data_asset_name="reddit_posts_asset",
                runtime_parameters={"batch_data": df},
                batch_identifiers={"default_identifier_name": "default"},
            )

            # Create validator
            validator = context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=suite_name,
            )

            # Add expectations
            validator.expect_column_values_to_not_be_null("Title")
            validator.expect_column_value_lengths_to_be_between("Title", 5, 300)
            validator.expect_column_values_to_match_regex("Title", r"^[A-Za-z0-9 .,!?'\-()]+$")
            validator.expect_column_values_to_match_regex("Title", r".*[A-Za-z].*")

            # Run validation
            results = validator.validate()
            failed_indices = set()
            for res in results.results:
                failed_indices.update(res.result.get("unexpected_index_list", []))

            # Drop failed rows and save cleaned table
            df_cleaned = df.drop(index=failed_indices) if failed_indices else df
            cleaned_count = len(df_cleaned)

            df_cleaned.to_sql(table_name, conn, if_exists='replace', index=False)
            print(f"✅ Cleaned and saved to DB. {cleaned_count} / {original_count} records kept.")

        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            if conn:
                conn.close()

    if __name__ == "__main__":
        clean_reddit_titles_from_db()

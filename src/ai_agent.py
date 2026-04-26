import re


def _extract_decline_rate(user_query):
    match = re.search(r"(\d+)\s*%?\s*(?:steeper\s+)?decline", user_query.lower())
    if match:
        return float(match.group(1))
    return None


def answer_analyst_query(user_query, forecast_df, selected_year):
    """Rule-based conversational analyst grounded in current dataframe."""
    q = user_query.strip().lower()
    if forecast_df.empty:
        return {
            "response": "No data is loaded yet. Add your EIA API key and refresh.",
            "actions": {},
            "data_backed": [],
            "inference": [],
        }

    ranked = forecast_df.sort_values("Projected_Production", ascending=False).reset_index(drop=True)
    top = ranked.iloc[0]
    actions = {}
    data_backed = []
    inference = []

    if "highest projected" in q or "top region" in q or "best region" in q:
        data_backed.append(
            f"{top['Region']} has the highest projected production for {selected_year}: "
            f"{int(top['Projected_Production']):,} bbl."
        )
        actions["selected_region"] = top["Region"]
        inference.append(
            f"Given this leadership position, {top['Region']} is a priority region for BD screening."
        )
    elif "summarize" in q or "opportunity" in q:
        region_match = next((r for r in ranked["Region"].tolist() if r.lower() in q), None)
        if region_match is None:
            region_match = top["Region"]
        row = ranked[ranked["Region"] == region_match].iloc[0]
        percentile = (ranked["Projected_Production"] <= row["Projected_Production"]).mean() * 100
        data_backed.append(
            f"{region_match} projected production ({selected_year}) is {int(row['Projected_Production']):,} bbl."
        )
        data_backed.append(
            f"{region_match} ranks in the top {max(1, int(100 - percentile))}% of tracked regions."
        )
        actions["selected_region"] = region_match
        inference.append(
            "Opportunity is stronger when high output aligns with stable trend and manageable decline assumptions."
        )
    elif "decline" in q or "what happens" in q:
        decline = _extract_decline_rate(user_query) or 15.0
        actions["decline_rate"] = decline
        data_backed.append(f"Scenario requested: apply {decline:.1f}% steeper annual decline to projections.")
        inference.append("Use stressed projections to compare downside resilience before capital allocation.")
    else:
        median_prod = int(ranked["Projected_Production"].median())
        data_backed.append(
            f"Current leader for {selected_year}: {top['Region']} ({int(top['Projected_Production']):,} bbl)."
        )
        data_backed.append(f"Median regional projected production: {median_prod:,} bbl.")
        inference.append("Ask for a specific region summary or decline stress test for decision-ready guidance.")

    response = "### Data-backed findings\n"
    response += "\n".join([f"- {point}" for point in data_backed])
    response += "\n\n### Model inference\n"
    response += "\n".join([f"- {point}" for point in inference])

    return {
        "response": response,
        "actions": actions,
        "data_backed": data_backed,
        "inference": inference,
    }
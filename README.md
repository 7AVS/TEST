"""
Phase 1: Universe Overview (40,000 feet view)
==============================================
Purpose: Establish baseline metrics for VVD campaign analysis
Output: Simple HTML tables for Excel/PowerPoint transfer
Author: Campaign Analysis Team
Date: July 2025

This script provides high-level baseline metrics without deep dives.
All metrics are at the aggregate level to understand the overall landscape.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from datetime import datetime

def generate_phase1_analysis(df):
    """
    Main function to generate Phase 1 baseline metrics
    
    Parameters:
    df: PySpark DataFrame with campaign data
    
    Returns:
    HTML formatted output with all baseline metrics
    """
    
    print("=" * 80)
    print("PHASE 1: UNIVERSE OVERVIEW - BASELINE METRICS")
    print("=" * 80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Data Source: /user/427966379/final_df.parquet")
    print("\n")
    
    # Start HTML output
    html_output = []
    html_output.append("<h1>Phase 1: Universe Overview - Baseline Metrics</h1>")
    html_output.append(f"<p>Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    
    # Filter to action group only for most metrics
    df_action = df.filter(F.col("TST_GRP_CD") == "TG4")
    
    # ========================================================================
    # 1. UNIVERSE SIZE METRICS
    # ========================================================================
    print("\n1. UNIVERSE SIZE METRICS")
    print("-" * 50)
    
    # Total deployments and unique clients (action group)
    total_deployments = df_action.count()
    unique_clients = df_action.select("CLNT_NO").distinct().count()
    
    # Total campaigns
    campaigns = df_action.select("MNE").distinct().collect()
    campaign_list = [row.MNE for row in campaigns]
    num_campaigns = len(campaign_list)
    
    # Time period
    date_range = df_action.select(
        F.min("TREATMT_STRT_DT").alias("min_date"),
        F.max("TREATMT_STRT_DT").alias("max_date")
    ).collect()[0]
    
    # Calculate months covered
    if date_range.min_date and date_range.max_date:
        months_covered = ((date_range.max_date.year - date_range.min_date.year) * 12 + 
                         date_range.max_date.month - date_range.min_date.month + 1)
    else:
        months_covered = 0
    
    print(f"Total Deployments (Action Group): {total_deployments:,}")
    print(f"Unique Clients (Action Group): {unique_clients:,}")
    print(f"Average Deployments per Client: {total_deployments/unique_clients:.2f}")
    print(f"Number of Campaigns: {num_campaigns}")
    print(f"Campaign List: {', '.join(campaign_list)}")
    print(f"Time Period: {date_range.min_date} to {date_range.max_date}")
    print(f"Months Covered: {months_covered}")
    
    # HTML Table 1
    html_output.append("<h2>1. Universe Size Metrics</h2>")
    html_output.append("<table border='1'>")
    html_output.append("<tr><th>Metric</th><th>Value</th></tr>")
    html_output.append(f"<tr><td>Total Deployments (Action Group)</td><td>{total_deployments:,}</td></tr>")
    html_output.append(f"<tr><td>Unique Clients (Action Group)</td><td>{unique_clients:,}</td></tr>")
    html_output.append(f"<tr><td>Average Deployments per Client</td><td>{total_deployments/unique_clients:.2f}</td></tr>")
    html_output.append(f"<tr><td>Number of Campaigns</td><td>{num_campaigns}</td></tr>")
    html_output.append(f"<tr><td>Campaign List</td><td>{', '.join(campaign_list)}</td></tr>")
    html_output.append(f"<tr><td>Time Period</td><td>{date_range.min_date} to {date_range.max_date}</td></tr>")
    html_output.append(f"<tr><td>Months Covered</td><td>{months_covered}</td></tr>")
    html_output.append("</table>")
    
    # ========================================================================
    # 2. TEST GROUP SPLIT
    # ========================================================================
    print("\n2. TEST GROUP SPLIT")
    print("-" * 50)
    
    test_group_split = df.groupBy("TST_GRP_CD").agg(
        F.count("*").alias("deployments"),
        F.countDistinct("CLNT_NO").alias("unique_clients")
    ).orderBy("TST_GRP_CD").collect()
    
    print(f"{'Test Group':<15} {'Deployments':>15} {'Unique Clients':>15} {'% of Total':>10}")
    print("-" * 55)
    
    html_output.append("<h2>2. Test Group Split</h2>")
    html_output.append("<table border='1'>")
    html_output.append("<tr><th>Test Group</th><th>Deployments</th><th>Unique Clients</th><th>% of Deployments</th></tr>")
    
    total_all_deployments = sum(row.deployments for row in test_group_split)
    for row in test_group_split:
        pct = (row.deployments / total_all_deployments) * 100
        print(f"{row.TST_GRP_CD:<15} {row.deployments:>15,} {row.unique_clients:>15,} {pct:>9.1f}%")
        html_output.append(f"<tr><td>{row.TST_GRP_CD}</td><td>{row.deployments:,}</td><td>{row.unique_clients:,}</td><td>{pct:.1f}%</td></tr>")
    
    html_output.append("</table>")
    
    # ========================================================================
    # 3. CAMPAIGN VOLUME BASELINES
    # ========================================================================
    print("\n3. CAMPAIGN VOLUME BASELINES (Action Group Only)")
    print("-" * 50)
    
    campaign_volumes = df_action.groupBy("MNE").agg(
        F.count("*").alias("deployments"),
        F.countDistinct("CLNT_NO").alias("unique_clients")
    ).orderBy(F.desc("deployments")).collect()
    
    print(f"{'Campaign':<10} {'Deployments':>15} {'Unique Clients':>15} {'% of Total':>12} {'Avg/Client':>12}")
    print("-" * 65)
    
    html_output.append("<h2>3. Campaign Volume Baselines (Action Group Only)</h2>")
    html_output.append("<table border='1'>")
    html_output.append("<tr><th>Campaign</th><th>Deployments</th><th>Unique Clients</th><th>% of Total</th><th>Avg Contacts/Client</th></tr>")
    
    for row in campaign_volumes:
        pct = (row.deployments / total_deployments) * 100
        avg_per_client = row.deployments / row.unique_clients if row.unique_clients > 0 else 0
        print(f"{row.MNE:<10} {row.deployments:>15,} {row.unique_clients:>15,} {pct:>11.1f}% {avg_per_client:>11.2f}")
        html_output.append(f"<tr><td>{row.MNE}</td><td>{row.deployments:,}</td><td>{row.unique_clients:,}</td><td>{pct:.1f}%</td><td>{avg_per_client:.2f}</td></tr>")
    
    html_output.append("</table>")
    
    # ========================================================================
    # 4. CONTACT FREQUENCY BASELINES
    # ========================================================================
    print("\n4. CONTACT FREQUENCY BASELINES (Action Group Only)")
    print("-" * 50)
    
    # Overall contact frequency distribution
    client_contacts = df_action.groupBy("CLNT_NO").agg(
        F.count("*").alias("total_contacts")
    )
    
    # Create frequency buckets
    freq_buckets = client_contacts.select(
        F.when(F.col("total_contacts") <= 3, "1-3 contacts")
        .when(F.col("total_contacts") <= 10, "4-10 contacts")
        .when(F.col("total_contacts") > 10, "11+ contacts")
        .alias("frequency_bucket")
    ).groupBy("frequency_bucket").count().collect()
    
    print("Overall Contact Distribution:")
    print(f"{'Frequency Bucket':<20} {'Clients':>12} {'% of Total':>12}")
    print("-" * 45)
    
    html_output.append("<h2>4. Contact Frequency Baselines (Action Group Only)</h2>")
    html_output.append("<h3>Overall Contact Distribution</h3>")
    html_output.append("<table border='1'>")
    html_output.append("<tr><th>Frequency Bucket</th><th>Clients</th><th>% of Total</th></tr>")
    
    # Sort buckets for consistent display
    bucket_order = ["1-3 contacts", "4-10 contacts", "11+ contacts"]
    freq_dict = {row.frequency_bucket: row.count for row in freq_buckets}
    
    for bucket in bucket_order:
        if bucket in freq_dict:
            count = freq_dict[bucket]
            pct = (count / unique_clients) * 100
            print(f"{bucket:<20} {count:>12,} {pct:>11.1f}%")
            html_output.append(f"<tr><td>{bucket}</td><td>{count:,}</td><td>{pct:.1f}%</td></tr>")
    
    html_output.append("</table>")
    
    # Monthly contact rate
    if months_covered > 0:
        monthly_rate = total_deployments / (unique_clients * months_covered)
        print(f"\nOverall Monthly Contact Rate: {monthly_rate:.2f} contacts/client/month")
        
        html_output.append("<h3>Monthly Contact Rate</h3>")
        html_output.append("<table border='1'>")
        html_output.append("<tr><th>Metric</th><th>Value</th></tr>")
        html_output.append(f"<tr><td>Overall Monthly Contact Rate</td><td>{monthly_rate:.2f} contacts/client/month</td></tr>")
        html_output.append("</table>")
    
    # ========================================================================
    # 5. SUCCESS RATE BASELINES
    # ========================================================================
    print("\n5. SUCCESS RATE BASELINES")
    print("-" * 50)
    
    # Define success metrics by campaign
    campaign_success_map = {
        'VCN': 'acquisition_success',
        'VDA': 'activation_success',  # Note: VDA is complex - acquisition but measured as activation
        'VDT': 'activation_success',
        'VUI': 'usage_success',
        'VAW': 'provisioning_success',
        'VUT': 'provisioning_success'
    }
    
    # Overall success rate across all campaigns
    success_columns = ['acquisition_success', 'activation_success', 'usage_success', 'provisioning_success']
    
    # Calculate overall success (any success in any campaign)
    overall_success_action = df_action.select(
        F.max(F.greatest(*[F.col(col) for col in success_columns if col in df_action.columns])).alias("any_success")
    ).agg(
        F.avg("any_success").alias("overall_success_rate")
    ).collect()[0].overall_success_rate or 0
    
    overall_success_control = df.filter(F.col("TST_GRP_CD") == "TG7").select(
        F.max(F.greatest(*[F.col(col) for col in success_columns if col in df.columns])).alias("any_success")
    ).agg(
        F.avg("any_success").alias("overall_success_rate")
    ).collect()[0].overall_success_rate or 0
    
    print(f"Overall Success Rate (Action Group TG4): {overall_success_action*100:.2f}%")
    print(f"Overall Success Rate (Control Group TG7): {overall_success_control*100:.2f}%")
    print(f"Overall Lift: {(overall_success_action - overall_success_control)*100:.2f}%")
    
    html_output.append("<h2>5. Success Rate Baselines</h2>")
    html_output.append("<table border='1'>")
    html_output.append("<tr><th>Metric</th><th>Rate</th></tr>")
    html_output.append(f"<tr><td>Overall Success Rate (Action Group TG4)</td><td>{overall_success_action*100:.2f}%</td></tr>")
    html_output.append(f"<tr><td>Overall Success Rate (Control Group TG7)</td><td>{overall_success_control*100:.2f}%</td></tr>")
    html_output.append(f"<tr><td>Overall Lift</td><td>{(overall_success_action - overall_success_control)*100:.2f}%</td></tr>")
    html_output.append("</table>")
    
    # Success by campaign type
    print("\nSuccess Rates by Campaign Type:")
    print(f"{'Type':<15} {'Action Rate':>12} {'Control Rate':>12} {'Lift':>10}")
    print("-" * 50)
    
    html_output.append("<h3>Success Rates by Campaign Type</h3>")
    html_output.append("<table border='1'>")
    html_output.append("<tr><th>Campaign Type</th><th>Action Rate</th><th>Control Rate</th><th>Lift</th></tr>")
    
    # Group by campaign goal type
    campaign_types = {
        'Acquisition': ['VCN'],
        'Activation': ['VDA', 'VDT'],
        'Usage': ['VUI'],
        'Provisioning': ['VAW', 'VUT']
    }
    
    for campaign_type, campaigns in campaign_types.items():
        # Get the appropriate success column
        if campaign_type == 'Acquisition':
            success_col = 'acquisition_success'
        elif campaign_type == 'Activation':
            success_col = 'activation_success'
        elif campaign_type == 'Usage':
            success_col = 'usage_success'
        else:  # Provisioning
            success_col = 'provisioning_success'
        
        if success_col in df.columns:
            action_rate = df_action.filter(F.col("MNE").isin(campaigns)).agg(
                F.avg(F.col(success_col)).alias("rate")
            ).collect()[0].rate or 0
            
            control_rate = df.filter((F.col("TST_GRP_CD") == "TG7") & (F.col("MNE").isin(campaigns))).agg(
                F.avg(F.col(success_col)).alias("rate")
            ).collect()[0].rate or 0
            
            lift = action_rate - control_rate
            
            print(f"{campaign_type:<15} {action_rate*100:>11.2f}% {control_rate*100:>11.2f}% {lift*100:>9.2f}%")
            html_output.append(f"<tr><td>{campaign_type}</td><td>{action_rate*100:.2f}%</td><td>{control_rate*100:.2f}%</td><td>{lift*100:.2f}%</td></tr>")
    
    html_output.append("</table>")
    
    # ========================================================================
    # 6. KEY OBSERVATIONS (AUTOMATED)
    # ========================================================================
    print("\n6. KEY OBSERVATIONS")
    print("-" * 50)
    
    html_output.append("<h2>6. Key Baseline Observations</h2>")
    html_output.append("<ul>")
    
    # Observation 1: Campaign dominance
    top_campaign = campaign_volumes[0]
    if top_campaign.deployments / total_deployments > 0.5:
        obs1 = f"{top_campaign.MNE} dominates with {(top_campaign.deployments/total_deployments)*100:.1f}% of all deployments"
        print(f"- {obs1}")
        html_output.append(f"<li>{obs1}</li>")
    
    # Observation 2: Over-contacting
    over_contacted = next((row for row in freq_buckets if row.frequency_bucket == "11+ contacts"), None)
    if over_contacted:
        obs2_pct = (over_contacted.count / unique_clients) * 100
        obs2 = f"{obs2_pct:.1f}% of clients receive 11+ contacts (potential over-contacting)"
        print(f"- {obs2}")
        html_output.append(f"<li>{obs2}</li>")
    
    # Observation 3: Monthly contact rate
    if months_covered > 0 and monthly_rate > 1:
        obs3 = f"Average client receives {monthly_rate:.1f} contacts per month"
        print(f"- {obs3}")
        html_output.append(f"<li>{obs3}</li>")
    
    # Observation 4: Test/Control split
    action_pct = next((row for row in test_group_split if row.TST_GRP_CD == "TG4"), None)
    if action_pct:
        obs4_pct = (action_pct.deployments / total_all_deployments) * 100
        obs4 = f"Action group (TG4) represents {obs4_pct:.1f}% of total deployments"
        print(f"- {obs4}")
        html_output.append(f"<li>{obs4}</li>")
    
    html_output.append("</ul>")
    
    # ========================================================================
    # GENERATE FINAL HTML OUTPUT
    # ========================================================================
    final_html = "\n".join(html_output)
    
    print("\n" + "=" * 80)
    print("HTML OUTPUT GENERATED - Ready for copy/paste to Excel or PowerPoint")
    print("=" * 80)
    
    return final_html


def save_html_output(html_content, filename="phase1_output.html"):
    """Save HTML output to file for easy access"""
    with open(filename, 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>VVD Campaign Analysis - Phase 1</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        h1 { color: #333; }
        h2 { color: #555; }
        h3 { color: #777; }
    </style>
</head>
<body>
""")
        f.write(html_content)
        f.write("""
</body>
</html>
""")
    print(f"\nHTML output saved to: {filename}")


# ========================================================================
# MAIN EXECUTION
# ========================================================================
if __name__ == "__main__":
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("VVD_Campaign_Analysis_Phase1") \
        .enableHiveSupport() \
        .getOrCreate()
    
    # Load data
    print("Loading data from HDFS...")
    df = spark.read.parquet("/user/427966379/final_df.parquet")
    
    # Generate Phase 1 analysis
    html_output = generate_phase1_analysis(df)
    
    # Save HTML output
    save_html_output(html_output)
    
    # Also save to HDFS if needed
    # You can uncomment and modify this section based on your HDFS setup
    # import subprocess
    # subprocess.call(['hdfs', 'dfs', '-put', 'phase1_output.html', '/user/427966379/'])
    
    print("\nPhase 1 analysis complete!")

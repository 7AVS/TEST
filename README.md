"""
================================================================================
PHASE 1: UNIVERSE OVERVIEW (40,000 FEET VIEW)
================================================================================
Purpose: Establish baseline metrics for VVD campaign analysis
Output: HTML tables displayed directly in Jupyter notebook cells
Author: Campaign Analysis Team
Date: July 2025

IMPORTANT: This is an exploratory analysis. No external files are created.
           The executed notebook serves as the audit trail.
           
Data Source: /user/427966379/final_df.parquet
================================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from IPython.display import display, HTML
from datetime import datetime


def generate_phase1_analysis(df):
    """
    Main function to generate Phase 1 baseline metrics
    
    This function performs high-level analysis without deep dives,
    establishing baselines that will be referenced throughout the analysis.
    
    Parameters:
    -----------
    df : PySpark DataFrame
        The complete campaign dataset with all fields
    
    Returns:
    --------
    None (displays HTML tables directly in Jupyter notebook)
    """
    
    # ========================================================================
    # HEADER SECTION - Analysis identification
    # ========================================================================
    display(HTML("<h1>PHASE 1: UNIVERSE OVERVIEW - BASELINE METRICS</h1>"))
    display(HTML(f"<p><b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"))
    display(HTML("<p><b>Data Source:</b> /user/427966379/final_df.parquet</p>"))
    display(HTML("<hr>"))
    
    # ========================================================================
    # DATA PREPARATION - Filter to action group for most metrics
    # ========================================================================
    # IMPORTANT: We analyze primarily the ACTION GROUP (TG4) as they received
    # the actual campaign communications. Control group (TG7) is used only
    # for lift calculations.
    df_action = df.filter(F.col("TST_GRP_CD") == "TG4")
    
    
    # ========================================================================
    # SECTION 1: UNIVERSE SIZE METRICS
    # ========================================================================
    # This section establishes the scale of our analysis - how much data
    # we're working with and what time period it covers.
    
    display(HTML("<h2>1. UNIVERSE SIZE METRICS</h2>"))
    display(HTML("<p>Understanding the scale and scope of our campaign data</p>"))
    
    # Calculate total deployments (each row is one client-tactic combination)
    total_deployments = df_action.count()
    
    # Calculate unique clients (removing duplicates from multiple campaigns)
    unique_clients = df_action.select("CLNT_NO").distinct().count()
    
    # Calculate average deployments per client
    avg_deployments_per_client = total_deployments / unique_clients if unique_clients > 0 else 0
    
    # Get list of all campaigns in the dataset
    campaigns = df_action.select("MNE").distinct().orderBy("MNE").collect()
    campaign_list = [row.MNE for row in campaigns]
    num_campaigns = len(campaign_list)
    
    # Determine the time period covered by the data
    # TREATMT_STRT_DT is when the campaign was deployed to the client
    date_range = df_action.select(
        F.min("TREATMT_STRT_DT").alias("min_date"),
        F.max("TREATMT_STRT_DT").alias("max_date")
    ).collect()[0]
    
    # Calculate the number of months covered in our analysis period
    if date_range.min_date and date_range.max_date:
        months_covered = ((date_range.max_date.year - date_range.min_date.year) * 12 + 
                         date_range.max_date.month - date_range.min_date.month + 1)
    else:
        months_covered = 0
    
    # Build and display the HTML table
    html_table = """
    <table border='1' style='border-collapse: collapse;'>
    <tr style='background-color: #f2f2f2;'>
        <th style='padding: 8px; text-align: left;'>Metric</th>
        <th style='padding: 8px; text-align: right;'>Value</th>
    </tr>
    """
    
    # Add each metric as a row in the table
    html_table += f"<tr><td style='padding: 8px;'>Total Deployments (Action Group)</td><td style='padding: 8px; text-align: right;'>{total_deployments:,}</td></tr>"
    html_table += f"<tr><td style='padding: 8px;'>Unique Clients (Action Group)</td><td style='padding: 8px; text-align: right;'>{unique_clients:,}</td></tr>"
    html_table += f"<tr><td style='padding: 8px;'>Average Deployments per Client</td><td style='padding: 8px; text-align: right;'>{avg_deployments_per_client:.2f}</td></tr>"
    html_table += f"<tr><td style='padding: 8px;'>Number of Campaigns</td><td style='padding: 8px; text-align: right;'>{num_campaigns}</td></tr>"
    html_table += f"<tr><td style='padding: 8px;'>Campaign List</td><td style='padding: 8px; text-align: right;'>{', '.join(campaign_list)}</td></tr>"
    html_table += f"<tr><td style='padding: 8px;'>Time Period</td><td style='padding: 8px; text-align: right;'>{date_range.min_date} to {date_range.max_date}</td></tr>"
    html_table += f"<tr><td style='padding: 8px;'>Months Covered</td><td style='padding: 8px; text-align: right;'>{months_covered}</td></tr>"
    html_table += "</table>"
    
    display(HTML(html_table))
    
    
    # ========================================================================
    # SECTION 2: TEST GROUP SPLIT
    # ========================================================================
    # This section shows the distribution between action and control groups.
    # This is critical for understanding the validity of our lift calculations.
    
    display(HTML("<h2>2. TEST GROUP SPLIT</h2>"))
    display(HTML("<p>Distribution of deployments between test and control groups</p>"))
    
    # Count deployments and unique clients by test group
    # TST_GRP_CD identifies whether a client was in action (TG4) or control (TG7)
    test_group_split = df.groupBy("TST_GRP_CD").agg(
        F.count("*").alias("deployments"),
        F.countDistinct("CLNT_NO").alias("unique_clients")
    ).orderBy("TST_GRP_CD").collect()
    
    # Calculate total deployments across all test groups for percentage calculation
    total_all_deployments = sum(row.deployments for row in test_group_split)
    
    # Build the test group comparison table
    html_table = """
    <table border='1' style='border-collapse: collapse;'>
    <tr style='background-color: #f2f2f2;'>
        <th style='padding: 8px;'>Test Group</th>
        <th style='padding: 8px; text-align: right;'>Deployments</th>
        <th style='padding: 8px; text-align: right;'>Unique Clients</th>
        <th style='padding: 8px; text-align: right;'>% of Deployments</th>
    </tr>
    """
    
    # Add a row for each test group
    for row in test_group_split:
        pct = (row.deployments / total_all_deployments) * 100 if total_all_deployments > 0 else 0
        
        # Highlight the test group codes for clarity
        group_desc = "Action Group" if row.TST_GRP_CD == "TG4" else "Control Group"
        
        html_table += f"""
        <tr>
            <td style='padding: 8px;'>{row.TST_GRP_CD} ({group_desc})</td>
            <td style='padding: 8px; text-align: right;'>{row.deployments:,}</td>
            <td style='padding: 8px; text-align: right;'>{row.unique_clients:,}</td>
            <td style='padding: 8px; text-align: right;'>{pct:.1f}%</td>
        </tr>
        """
    
    html_table += "</table>"
    display(HTML(html_table))
    
    
    # ========================================================================
    # SECTION 3: CAMPAIGN VOLUME BASELINES
    # ========================================================================
    # This section shows the distribution of deployments across campaigns.
    # Key insight: Are some campaigns dominating the contact strategy?
    
    display(HTML("<h2>3. CAMPAIGN VOLUME BASELINES (Action Group Only)</h2>"))
    display(HTML("<p>Volume and reach metrics by campaign - identifying potential imbalances</p>"))
    
    # Aggregate deployments and unique clients by campaign (MNE)
    # This helps identify which campaigns are dominating customer contacts
    campaign_volumes = df_action.groupBy("MNE").agg(
        F.count("*").alias("deployments"),
        F.countDistinct("CLNT_NO").alias("unique_clients")
    ).orderBy(F.desc("deployments")).collect()
    
    # Build the campaign volume comparison table
    html_table = """
    <table border='1' style='border-collapse: collapse;'>
    <tr style='background-color: #f2f2f2;'>
        <th style='padding: 8px;'>Campaign</th>
        <th style='padding: 8px; text-align: right;'>Deployments</th>
        <th style='padding: 8px; text-align: right;'>Unique Clients</th>
        <th style='padding: 8px; text-align: right;'>% of Total</th>
        <th style='padding: 8px; text-align: right;'>Avg Contacts/Client</th>
    </tr>
    """
    
    # Add each campaign's metrics
    for row in campaign_volumes:
        # Calculate percentage of total deployments
        pct = (row.deployments / total_deployments) * 100 if total_deployments > 0 else 0
        
        # Calculate average contacts per client for this campaign
        avg_per_client = row.deployments / row.unique_clients if row.unique_clients > 0 else 0
        
        # Highlight if a campaign is dominating (>50% of deployments)
        row_style = " style='background-color: #ffe6e6;'" if pct > 50 else ""
        
        html_table += f"""
        <tr{row_style}>
            <td style='padding: 8px;'>{row.MNE}</td>
            <td style='padding: 8px; text-align: right;'>{row.deployments:,}</td>
            <td style='padding: 8px; text-align: right;'>{row.unique_clients:,}</td>
            <td style='padding: 8px; text-align: right;'>{pct:.1f}%</td>
            <td style='padding: 8px; text-align: right;'>{avg_per_client:.2f}</td>
        </tr>
        """
    
    html_table += "</table>"
    display(HTML(html_table))
    
    
    # ========================================================================
    # SECTION 4: CONTACT FREQUENCY BASELINES
    # ========================================================================
    # This section examines how many times clients are being contacted.
    # Key question: Are we over-contacting our clients?
    
    display(HTML("<h2>4. CONTACT FREQUENCY BASELINES (Action Group Only)</h2>"))
    display(HTML("<p>Understanding contact patterns - are we over-contacting clients?</p>"))
    
    # Calculate total contacts per client across ALL campaigns
    client_contacts = df_action.groupBy("CLNT_NO").agg(
        F.count("*").alias("total_contacts")
    )
    
    # Create frequency buckets to categorize contact levels
    # These buckets help identify over-contacted segments
    freq_buckets = client_contacts.select(
        F.when(F.col("total_contacts") <= 3, "1-3 contacts")
        .when(F.col("total_contacts") <= 10, "4-10 contacts")
        .when(F.col("total_contacts") > 10, "11+ contacts")
        .alias("frequency_bucket")
    ).groupBy("frequency_bucket").count().collect()
    
    # Create a dictionary for easier bucket ordering
    freq_dict = {row.frequency_bucket: row.count for row in freq_buckets}
    bucket_order = ["1-3 contacts", "4-10 contacts", "11+ contacts"]
    
    # Build the contact frequency distribution table
    display(HTML("<h3>Overall Contact Distribution</h3>"))
    
    html_table = """
    <table border='1' style='border-collapse: collapse;'>
    <tr style='background-color: #f2f2f2;'>
        <th style='padding: 8px;'>Frequency Bucket</th>
        <th style='padding: 8px; text-align: right;'>Number of Clients</th>
        <th style='padding: 8px; text-align: right;'>% of Total Clients</th>
    </tr>
    """
    
    # Add rows in logical order
    for bucket in bucket_order:
        if bucket in freq_dict:
            count = freq_dict[bucket]
            pct = (count / unique_clients) * 100 if unique_clients > 0 else 0
            
            # Highlight the over-contacted segment (11+ contacts)
            row_style = " style='background-color: #ffe6e6;'" if bucket == "11+ contacts" and pct > 20 else ""
            
            html_table += f"""
            <tr{row_style}>
                <td style='padding: 8px;'>{bucket}</td>
                <td style='padding: 8px; text-align: right;'>{count:,}</td>
                <td style='padding: 8px; text-align: right;'>{pct:.1f}%</td>
            </tr>
            """
    
    html_table += "</table>"
    display(HTML(html_table))
    
    # Calculate and display monthly contact rate
    # This helps understand contact velocity
    if months_covered > 0:
        monthly_rate = total_deployments / (unique_clients * months_covered)
        
        display(HTML("<h3>Monthly Contact Rate</h3>"))
        html_table = f"""
        <table border='1' style='border-collapse: collapse;'>
        <tr style='background-color: #f2f2f2;'>
            <th style='padding: 8px;'>Metric</th>
            <th style='padding: 8px; text-align: right;'>Value</th>
        </tr>
        <tr>
            <td style='padding: 8px;'>Overall Monthly Contact Rate</td>
            <td style='padding: 8px; text-align: right;'>{monthly_rate:.2f} contacts/client/month</td>
        </tr>
        </table>
        """
        display(HTML(html_table))
    
    
    # ========================================================================
    # SECTION 5: SUCCESS RATE BASELINES
    # ========================================================================
    # This section establishes baseline success rates for lift calculations.
    # We compare action group (TG4) vs control group (TG7).
    
    display(HTML("<h2>5. SUCCESS RATE BASELINES</h2>"))
    display(HTML("<p>Baseline success rates to understand campaign effectiveness</p>"))
    
    # Define which success metric applies to each campaign
    # This mapping is critical for accurate success measurement
    campaign_success_map = {
        'VCN': 'acquisition_success',    # Acquisition campaign
        'VDA': 'activation_success',      # Activation campaign (note: data quality issue)
        'VDT': 'activation_success',      # Activation campaign
        'VUI': 'usage_success',           # Usage campaign
        'VAW': 'provisioning_success',    # Provisioning campaign
        'VUT': 'provisioning_success'     # Provisioning campaign
    }
    
    # List all possible success columns
    success_columns = ['acquisition_success', 'activation_success', 'usage_success', 'provisioning_success']
    
    # Filter to only columns that exist in the dataframe
    existing_success_cols = [col for col in success_columns if col in df.columns]
    
    # Calculate overall success rate (any success in any campaign)
    # Using greatest() to find if client had ANY success
    if existing_success_cols:
        # Action group success rate
        overall_success_action = df_action.select(
            F.max(F.greatest(*[F.col(col) for col in existing_success_cols])).alias("any_success")
        ).agg(
            F.avg("any_success").alias("overall_success_rate")
        ).collect()[0].overall_success_rate or 0
        
        # Control group success rate (natural rate without campaigns)
        overall_success_control = df.filter(F.col("TST_GRP_CD") == "TG7").select(
            F.max(F.greatest(*[F.col(col) for col in existing_success_cols])).alias("any_success")
        ).agg(
            F.avg("any_success").alias("overall_success_rate")
        ).collect()[0].overall_success_rate or 0
        
        # Calculate lift (incremental impact of campaigns)
        overall_lift = overall_success_action - overall_success_control
    else:
        overall_success_action = 0
        overall_success_control = 0
        overall_lift = 0
    
    # Display overall success metrics
    html_table = """
    <table border='1' style='border-collapse: collapse;'>
    <tr style='background-color: #f2f2f2;'>
        <th style='padding: 8px;'>Metric</th>
        <th style='padding: 8px; text-align: right;'>Rate</th>
    </tr>
    """
    
    html_table += f"""
    <tr>
        <td style='padding: 8px;'>Overall Success Rate (Action Group TG4)</td>
        <td style='padding: 8px; text-align: right;'>{overall_success_action*100:.2f}%</td>
    </tr>
    <tr>
        <td style='padding: 8px;'>Overall Success Rate (Control Group TG7)</td>
        <td style='padding: 8px; text-align: right;'>{overall_success_control*100:.2f}%</td>
    </tr>
    <tr>
        <td style='padding: 8px;'>Overall Lift (Absolute)</td>
        <td style='padding: 8px; text-align: right;'>{overall_lift*100:.2f}%</td>
    </tr>
    """
    
    html_table += "</table>"
    display(HTML(html_table))
    
    # Success rates by campaign type
    # Group campaigns by their objective
    display(HTML("<h3>Success Rates by Campaign Type</h3>"))
    
    campaign_types = {
        'Acquisition': ['VCN'],
        'Activation': ['VDA', 'VDT'],
        'Usage': ['VUI'],
        'Provisioning': ['VAW', 'VUT']
    }
    
    html_table = """
    <table border='1' style='border-collapse: collapse;'>
    <tr style='background-color: #f2f2f2;'>
        <th style='padding: 8px;'>Campaign Type</th>
        <th style='padding: 8px; text-align: right;'>Action Rate</th>
        <th style='padding: 8px; text-align: right;'>Control Rate</th>
        <th style='padding: 8px; text-align: right;'>Lift (Absolute)</th>
    </tr>
    """
    
    # Calculate success rates for each campaign type
    for campaign_type, campaigns in campaign_types.items():
        # Determine the appropriate success column for this campaign type
        if campaign_type == 'Acquisition':
            success_col = 'acquisition_success'
        elif campaign_type == 'Activation':
            success_col = 'activation_success'
        elif campaign_type == 'Usage':
            success_col = 'usage_success'
        else:  # Provisioning
            success_col = 'provisioning_success'
        
        # Only calculate if the success column exists in the data
        if success_col in df.columns:
            # Action group rate for this campaign type
            action_rate = df_action.filter(F.col("MNE").isin(campaigns)).agg(
                F.avg(F.col(success_col)).alias("rate")
            ).collect()[0].rate or 0
            
            # Control group rate for this campaign type
            control_rate = df.filter(
                (F.col("TST_GRP_CD") == "TG7") & 
                (F.col("MNE").isin(campaigns))
            ).agg(
                F.avg(F.col(success_col)).alias("rate")
            ).collect()[0].rate or 0
            
            # Calculate absolute lift
            lift = action_rate - control_rate
            
            html_table += f"""
            <tr>
                <td style='padding: 8px;'>{campaign_type}</td>
                <td style='padding: 8px; text-align: right;'>{action_rate*100:.2f}%</td>
                <td style='padding: 8px; text-align: right;'>{control_rate*100:.2f}%</td>
                <td style='padding: 8px; text-align: right;'>{lift*100:.2f}%</td>
            </tr>
            """
    
    html_table += "</table>"
    display(HTML(html_table))
    
    
    # ========================================================================
    # SECTION 6: KEY BASELINE OBSERVATIONS (AUTOMATED)
    # ========================================================================
    # This section automatically identifies notable patterns in the baseline data
    
    display(HTML("<h2>6. KEY BASELINE OBSERVATIONS</h2>"))
    display(HTML("<p>Automated insights from the baseline metrics</p>"))
    
    observations = []
    
    # Observation 1: Check for campaign dominance
    if campaign_volumes:
        top_campaign = campaign_volumes[0]
        dominance_pct = (top_campaign.deployments / total_deployments) * 100
        if dominance_pct > 50:
            observations.append(f"{top_campaign.MNE} dominates with {dominance_pct:.1f}% of all deployments")
    
    # Observation 2: Check for over-contacting
    if "11+ contacts" in freq_dict:
        over_contacted_pct = (freq_dict["11+ contacts"] / unique_clients) * 100
        if over_contacted_pct > 20:
            observations.append(f"{over_contacted_pct:.1f}% of clients receive 11+ contacts (potential over-contacting)")
    
    # Observation 3: Monthly contact rate insight
    if months_covered > 0:
        if monthly_rate > 1:
            observations.append(f"Average client receives {monthly_rate:.1f} contacts per month")
    
    # Observation 4: Test/Control split insight
    action_group = next((row for row in test_group_split if row.TST_GRP_CD == "TG4"), None)
    if action_group and total_all_deployments > 0:
        action_pct = (action_group.deployments / total_all_deployments) * 100
        observations.append(f"Action group (TG4) represents {action_pct:.1f}% of total deployments")
    
    # Display observations as a bulleted list
    if observations:
        obs_html = "<ul>"
        for obs in observations:
            obs_html += f"<li>{obs}</li>"
        obs_html += "</ul>"
        display(HTML(obs_html))
    else:
        display(HTML("<p>No significant patterns detected in baseline metrics.</p>"))
    
    
    # ========================================================================
    # FOOTER - Analysis completion confirmation
    # ========================================================================
    display(HTML("<hr>"))
    display(HTML("<p><b>Phase 1 Analysis Complete</b> - Baseline metrics established</p>"))


# ========================================================================
# MAIN EXECUTION BLOCK
# ========================================================================
if __name__ == "__main__":
    """
    To run this analysis:
    1. Ensure Spark session is initialized
    2. Load the data from HDFS
    3. Call generate_phase1_analysis(df)
    
    Example:
    --------
    spark = SparkSession.builder \
        .appName("VVD_Campaign_Analysis_Phase1") \
        .enableHiveSupport() \
        .getOrCreate()
    
    df = spark.read.parquet("/user/427966379/final_df.parquet")
    generate_phase1_analysis(df)
    """
    
    # Note: The actual execution will be done in the Jupyter notebook
    # This block is here for documentation purposes
    pass

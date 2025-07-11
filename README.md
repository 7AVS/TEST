"""
================================================================================
PHASE 2: CAMPAIGN (MNE) INTERACTION ANALYSIS
================================================================================
Purpose: Deep dive into individual campaign performance and interactions
Output: HTML tables displayed directly in Jupyter notebook cells
Author: Campaign Analysis Team
Date: July 2025
Version: 1.0

This phase analyzes:
1. Campaign-specific volume and reach metrics
2. Performance by contact frequency (diminishing returns)
3. Campaign overlap patterns
4. Temporal patterns and seasonality

IMPORTANT: Following PySpark best practices from Phase 1 v2
- Explicit type conversions for Row objects
- No Python built-in naming conflicts
- Clear distinction between PySpark and Python functions
           
Data Source: /user/427966379/final_df.parquet
================================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from IPython.display import display, HTML
from datetime import datetime
import math
# Import Python built-ins with different names to avoid conflicts
from builtins import sum as python_sum
from builtins import min as python_min
from builtins import max as python_max


def calculate_statistical_significance(success_a, total_a, success_b, total_b, alpha=0.05):
    """
    Calculate statistical significance using z-test for proportions
    (Copied from Phase 1 v2 for consistency)
    """
    # Avoid division by zero
    if total_a == 0 or total_b == 0:
        return {
            'p_value': 1.0,
            'is_significant': False,
            'confidence_interval': (0, 0),
            'z_score': 0
        }
    
    # Calculate proportions
    p_a = float(success_a) / float(total_a)
    p_b = float(success_b) / float(total_b)
    
    # Pooled proportion
    p_pooled = float(success_a + success_b) / float(total_a + total_b)
    
    # Standard error
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/float(total_a) + 1/float(total_b)))
    
    # Z-score
    if se == 0:
        z_score = 0
    else:
        z_score = (p_a - p_b) / se
    
    # P-value (two-tailed test)
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / math.sqrt(2))))
    
    # Confidence interval for the difference
    z_critical = 1.96  # for 95% confidence
    margin_of_error = z_critical * se
    ci_lower = (p_a - p_b) - margin_of_error
    ci_upper = (p_a - p_b) + margin_of_error
    
    return {
        'p_value': p_value,
        'is_significant': p_value < alpha,
        'confidence_interval': (ci_lower, ci_upper),
        'z_score': z_score
    }


def generate_phase2_analysis(df):
    """
    Main function to generate Phase 2 campaign interaction analysis
    
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
    display(HTML("<h1>PHASE 2: CAMPAIGN (MNE) INTERACTION ANALYSIS</h1>"))
    display(HTML(f"<p><b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"))
    display(HTML("<p><b>Data Source:</b> /user/427966379/final_df.parquet</p>"))
    display(HTML("<hr>"))
    
    # ========================================================================
    # DATA PREPARATION
    # ========================================================================
    # Filter to action group for main analysis
    df_action = df.filter(F.col("TST_GRP_CD") == "TG4")
    df_control = df.filter(F.col("TST_GRP_CD") == "TG7")
    
    # Get list of campaigns for iteration
    campaigns_list = df_action.select("MNE").distinct().orderBy("MNE").collect()
    campaign_names = [row.MNE for row in campaigns_list]
    
    # ========================================================================
    # SECTION 1: CAMPAIGN VOLUME AND REACH METRICS
    # ========================================================================
    display(HTML("<h2>1. CAMPAIGN VOLUME AND REACH METRICS</h2>"))
    display(HTML("<p>Detailed deployment patterns and client reach by campaign</p>"))
    
    # Calculate metrics for each campaign
    campaign_metrics = []
    
    for campaign in campaign_names:
        # Filter data for this campaign
        df_campaign_action = df_action.filter(F.col("MNE") == campaign)
        df_campaign_control = df_control.filter(F.col("MNE") == campaign)
        
        # Calculate deployment metrics
        action_deployments = df_campaign_action.count()
        control_deployments = df_campaign_control.count()
        
        # Calculate unique clients
        action_unique_clients = df_campaign_action.select("CLNT_NO").distinct().count()
        control_unique_clients = df_campaign_control.select("CLNT_NO").distinct().count()
        
        # Calculate contact frequency distribution for this campaign
        if action_deployments > 0:
            contact_freq = df_campaign_action.groupBy("CLNT_NO").agg(
                F.count("*").alias("contacts")
            ).groupBy("contacts").count().orderBy("contacts").collect()
            
            # Find max contacts using explicit loop
            max_contacts = 0
            for row in contact_freq:
                if int(row.contacts) > max_contacts:
                    max_contacts = int(row.contacts)
        else:
            max_contacts = 0
        
        # Store metrics
        campaign_metrics.append({
            'campaign': campaign,
            'action_deployments': action_deployments,
            'control_deployments': control_deployments,
            'action_unique_clients': action_unique_clients,
            'control_unique_clients': control_unique_clients,
            'avg_contacts_per_client': float(action_deployments) / float(action_unique_clients) if action_unique_clients > 0 else 0,
            'max_contacts': max_contacts
        })
    
    # Display campaign volume table
    html_table = """
    <table border='1' style='border-collapse: collapse;'>
    <tr style='background-color: #f2f2f2;'>
        <th style='padding: 8px;'>Campaign</th>
        <th style='padding: 8px; text-align: right;'>Action Deployments</th>
        <th style='padding: 8px; text-align: right;'>Control Deployments</th>
        <th style='padding: 8px; text-align: right;'>Action Unique Clients</th>
        <th style='padding: 8px; text-align: right;'>Control Unique Clients</th>
        <th style='padding: 8px; text-align: right;'>Avg Contacts/Client</th>
        <th style='padding: 8px; text-align: right;'>Max Contacts</th>
    </tr>
    """
    
    for metrics in campaign_metrics:
        html_table += f"""
        <tr>
            <td style='padding: 8px;'>{metrics['campaign']}</td>
            <td style='padding: 8px; text-align: right;'>{metrics['action_deployments']:,}</td>
            <td style='padding: 8px; text-align: right;'>{metrics['control_deployments']:,}</td>
            <td style='padding: 8px; text-align: right;'>{metrics['action_unique_clients']:,}</td>
            <td style='padding: 8px; text-align: right;'>{metrics['control_unique_clients']:,}</td>
            <td style='padding: 8px; text-align: right;'>{metrics['avg_contacts_per_client']:.2f}</td>
            <td style='padding: 8px; text-align: right;'>{metrics['max_contacts']}</td>
        </tr>
        """
    
    html_table += "</table>"
    display(HTML(html_table))
    
    # ========================================================================
    # SECTION 2: CAMPAIGN PERFORMANCE BY CONTACT FREQUENCY
    # ========================================================================
    display(HTML("<h2>2. CAMPAIGN PERFORMANCE BY CONTACT FREQUENCY</h2>"))
    display(HTML("<p>Diminishing returns analysis - success rates by number of contacts</p>"))
    
    # Define success metric mapping (from Phase 1)
    campaign_success_map = {
        'VCN': 'acquisition_success',
        'VDA': 'activation_success',
        'VDT': 'activation_success',
        'VUI': 'usage_success',
        'VAW': 'provisioning_success',
        'VUT': 'provisioning_success'
    }
    
    # Analyze each campaign's diminishing returns
    for campaign in campaign_names:
        display(HTML(f"<h3>Campaign: {campaign}</h3>"))
        
        # Get the appropriate success metric
        success_col = campaign_success_map.get(campaign, 'activation_success')
        
        # Check if success column exists
        if success_col not in df.columns:
            display(HTML(f"<p style='color: red;'>Success column '{success_col}' not found for campaign {campaign}</p>"))
            continue
        
        # Create contact sequence number using window function
        window_spec = Window.partitionBy("CLNT_NO").orderBy("TREATMT_STRT_DT")
        
        # Add contact sequence number
        df_campaign_seq = df_action.filter(F.col("MNE") == campaign).withColumn(
            "contact_sequence", F.row_number().over(window_spec)
        )
        
        # Create contact buckets
        df_campaign_buckets = df_campaign_seq.withColumn(
            "contact_bucket",
            F.when(F.col("contact_sequence") == 1, "1")
            .when(F.col("contact_sequence") == 2, "2")
            .when(F.col("contact_sequence") == 3, "3")
            .when((F.col("contact_sequence") >= 4) & (F.col("contact_sequence") <= 5), "4-5")
            .when((F.col("contact_sequence") >= 6) & (F.col("contact_sequence") <= 10), "6-10")
            .otherwise("11+")
        )
        
        # Calculate performance by bucket
        performance_by_bucket = df_campaign_buckets.groupBy("contact_bucket").agg(
            F.count("*").alias("deployments"),
            F.countDistinct("CLNT_NO").alias("unique_clients"),
            F.sum(F.when(F.col(success_col) == 1, 1).otherwise(0)).alias("successes")
        ).orderBy("contact_bucket").collect()
        
        # Display performance table
        html_table = """
        <table border='1' style='border-collapse: collapse;'>
        <tr style='background-color: #f2f2f2;'>
            <th style='padding: 8px;'>Contact Bucket</th>
            <th style='padding: 8px; text-align: right;'>Deployments</th>
            <th style='padding: 8px; text-align: right;'>Unique Clients</th>
            <th style='padding: 8px; text-align: right;'>Successes</th>
            <th style='padding: 8px; text-align: right;'>Success Rate</th>
            <th style='padding: 8px; text-align: right;'>Incremental Change</th>
        </tr>
        """
        
        # Track previous success rate for incremental calculation
        prev_success_rate = None
        bucket_order = ["1", "2", "3", "4-5", "6-10", "11+"]
        
        # Create dict for ordered display
        bucket_dict = {}
        for row in performance_by_bucket:
            bucket_dict[row.contact_bucket] = row
        
        for bucket in bucket_order:
            if bucket in bucket_dict:
                row = bucket_dict[bucket]
                deployments = int(row.deployments)
                unique_clients = int(row.unique_clients)
                successes = int(row.successes)
                success_rate = float(successes) / float(deployments) if deployments > 0 else 0
                
                # Calculate incremental change
                if prev_success_rate is not None:
                    incremental = success_rate - prev_success_rate
                    incremental_str = f"{incremental*100:+.2f}%"
                else:
                    incremental_str = "N/A"
                
                # Highlight declining performance
                row_style = " style='background-color: #ffe6e6;'" if prev_success_rate and success_rate < prev_success_rate else ""
                
                html_table += f"""
                <tr{row_style}>
                    <td style='padding: 8px;'>{bucket}</td>
                    <td style='padding: 8px; text-align: right;'>{deployments:,}</td>
                    <td style='padding: 8px; text-align: right;'>{unique_clients:,}</td>
                    <td style='padding: 8px; text-align: right;'>{successes:,}</td>
                    <td style='padding: 8px; text-align: right;'>{success_rate*100:.2f}%</td>
                    <td style='padding: 8px; text-align: right;'>{incremental_str}</td>
                </tr>
                """
                
                prev_success_rate = success_rate
        
        html_table += "</table>"
        display(HTML(html_table))
    
    # ========================================================================
    # SECTION 3: CAMPAIGN OVERLAP ANALYSIS
    # ========================================================================
    display(HTML("<h2>3. CAMPAIGN OVERLAP ANALYSIS</h2>"))
    display(HTML("<p>Identifying clients receiving multiple campaigns within 25-day windows</p>"))
    
    # Self-join to find overlaps
    df1 = df_action.select(
        F.col("CLNT_NO").alias("client"),
        F.col("MNE").alias("campaign1"),
        F.col("TREATMT_STRT_DT").alias("start_date1")
    )
    
    df2 = df_action.select(
        F.col("CLNT_NO").alias("client"),
        F.col("MNE").alias("campaign2"),
        F.col("TREATMT_STRT_DT").alias("start_date2")
    )
    
    # Join and calculate date differences
    overlaps = df1.join(
        df2,
        (df1.client == df2.client) & 
        (df1.start_date1 < df2.start_date2)
    ).withColumn(
        "days_between", 
        F.datediff(F.col("start_date2"), F.col("start_date1"))
    ).filter(
        F.col("days_between") <= 25
    )
    
    # Count overlaps by campaign pair
    overlap_summary = overlaps.groupBy("campaign1", "campaign2").agg(
        F.countDistinct("client").alias("clients_affected"),
        F.count("*").alias("overlap_instances"),
        F.avg("days_between").alias("avg_days_between")
    ).orderBy(F.desc("clients_affected")).collect()
    
    # Display overlap matrix
    display(HTML("<h3>Campaign Overlap Matrix (≤25 days)</h3>"))
    
    html_table = """
    <table border='1' style='border-collapse: collapse;'>
    <tr style='background-color: #f2f2f2;'>
        <th style='padding: 8px;'>Campaign 1</th>
        <th style='padding: 8px;'>Campaign 2</th>
        <th style='padding: 8px; text-align: right;'>Clients Affected</th>
        <th style='padding: 8px; text-align: right;'>Overlap Instances</th>
        <th style='padding: 8px; text-align: right;'>Avg Days Between</th>
    </tr>
    """
    
    # Display top overlaps
    overlap_count = 0
    for row in overlap_summary:
        if overlap_count >= 20:  # Limit to top 20 for readability
            break
        
        # Highlight self-overlaps
        is_self_overlap = row.campaign1 == row.campaign2
        row_style = " style='background-color: #ffcccc;'" if is_self_overlap else ""
        
        html_table += f"""
        <tr{row_style}>
            <td style='padding: 8px;'>{row.campaign1}</td>
            <td style='padding: 8px;'>{row.campaign2}</td>
            <td style='padding: 8px; text-align: right;'>{int(row.clients_affected):,}</td>
            <td style='padding: 8px; text-align: right;'>{int(row.overlap_instances):,}</td>
            <td style='padding: 8px; text-align: right;'>{float(row.avg_days_between):.1f}</td>
        </tr>
        """
        overlap_count += 1
    
    html_table += "</table>"
    display(HTML(html_table))
    
    # Calculate total clients with any overlap
    total_overlap_clients = overlaps.select("client").distinct().count()
    display(HTML(f"<p><b>Total unique clients with campaign overlaps (≤25 days):</b> {total_overlap_clients:,}</p>"))
    
    # ========================================================================
    # SECTION 4: CAMPAIGN TEMPORAL PATTERNS
    # ========================================================================
    display(HTML("<h2>4. CAMPAIGN TEMPORAL PATTERNS</h2>"))
    display(HTML("<p>Monthly deployment trends and seasonality by campaign</p>"))
    
    # Add month column
    df_monthly = df_action.withColumn(
        "year_month", 
        F.date_format("TREATMT_STRT_DT", "yyyy-MM")
    )
    
    # Calculate monthly deployments by campaign
    monthly_trends = df_monthly.groupBy("year_month", "MNE").agg(
        F.count("*").alias("deployments"),
        F.countDistinct("CLNT_NO").alias("unique_clients")
    ).orderBy("year_month", "MNE").collect()
    
    # Organize data by campaign
    campaign_monthly_data = {}
    for row in monthly_trends:
        campaign = row.MNE
        if campaign not in campaign_monthly_data:
            campaign_monthly_data[campaign] = []
        campaign_monthly_data[campaign].append({
            'month': row.year_month,
            'deployments': int(row.deployments),
            'unique_clients': int(row.unique_clients)
        })
    
    # Display trends for each campaign
    for campaign in sorted(campaign_monthly_data.keys()):
        display(HTML(f"<h3>Campaign {campaign} - Monthly Trends</h3>"))
        
        months_data = campaign_monthly_data[campaign]
        
        # Calculate campaign statistics
        deployment_values = [m['deployments'] for m in months_data]
        
        # Calculate min/max/avg using explicit loops
        min_deployments = deployment_values[0] if deployment_values else 0
        max_deployments = deployment_values[0] if deployment_values else 0
        total_deployments = 0
        
        for val in deployment_values:
            if val < min_deployments:
                min_deployments = val
            if val > max_deployments:
                max_deployments = val
            total_deployments += val
        
        avg_deployments = float(total_deployments) / float(len(deployment_values)) if deployment_values else 0
        
        # Check for seasonality (high variance indicates seasonal pattern)
        variance = 0
        if len(deployment_values) > 1:
            for val in deployment_values:
                variance += (float(val) - avg_deployments) ** 2
            variance = variance / float(len(deployment_values))
            std_dev = math.sqrt(variance)
            cv = std_dev / avg_deployments if avg_deployments > 0 else 0
        else:
            cv = 0
        
        seasonality = "High" if cv > 0.5 else "Low"
        
        # Display summary stats
        html_table = f"""
        <table border='1' style='border-collapse: collapse;'>
        <tr style='background-color: #f2f2f2;'>
            <th style='padding: 8px;'>Metric</th>
            <th style='padding: 8px; text-align: right;'>Value</th>
        </tr>
        <tr>
            <td style='padding: 8px;'>Active Months</td>
            <td style='padding: 8px; text-align: right;'>{len(months_data)}</td>
        </tr>
        <tr>
            <td style='padding: 8px;'>Min Monthly Deployments</td>
            <td style='padding: 8px; text-align: right;'>{min_deployments:,}</td>
        </tr>
        <tr>
            <td style='padding: 8px;'>Max Monthly Deployments</td>
            <td style='padding: 8px; text-align: right;'>{max_deployments:,}</td>
        </tr>
        <tr>
            <td style='padding: 8px;'>Avg Monthly Deployments</td>
            <td style='padding: 8px; text-align: right;'>{avg_deployments:,.0f}</td>
        </tr>
        <tr>
            <td style='padding: 8px;'>Seasonality Indicator</td>
            <td style='padding: 8px; text-align: right;'>{seasonality} (CV: {cv:.2f})</td>
        </tr>
        </table>
        """
        display(HTML(html_table))
    
    # ========================================================================
    # SECTION 5: CAMPAIGN SUCCESS COMPARISON
    # ========================================================================
    display(HTML("<h2>5. CAMPAIGN SUCCESS COMPARISON</h2>"))
    display(HTML("<p>Head-to-head campaign performance with statistical significance</p>"))
    
    html_table = """
    <table border='1' style='border-collapse: collapse;'>
    <tr style='background-color: #f2f2f2;'>
        <th style='padding: 8px;'>Campaign</th>
        <th style='padding: 8px; text-align: right;'>Action Success Rate</th>
        <th style='padding: 8px; text-align: right;'>Control Success Rate</th>
        <th style='padding: 8px; text-align: right;'>Absolute Lift</th>
        <th style='padding: 8px; text-align: right;'>Relative Lift</th>
        <th style='padding: 8px; text-align: right;'>P-Value</th>
        <th style='padding: 8px;'>Significant?</th>
    </tr>
    """
    
    for campaign in campaign_names:
        # Get success metric for this campaign
        success_col = campaign_success_map.get(campaign, 'activation_success')
        
        if success_col not in df.columns:
            continue
        
        # Calculate success metrics for action group
        action_metrics = df_action.filter(F.col("MNE") == campaign).agg(
            F.count("*").alias("total"),
            F.sum(F.when(F.col(success_col) == 1, 1).otherwise(0)).alias("successes")
        ).collect()[0]
        
        action_total = int(action_metrics.total) if action_metrics.total else 0
        action_success = int(action_metrics.successes) if action_metrics.successes else 0
        action_rate = float(action_success) / float(action_total) if action_total > 0 else 0
        
        # Calculate success metrics for control group
        control_metrics = df_control.filter(F.col("MNE") == campaign).agg(
            F.count("*").alias("total"),
            F.sum(F.when(F.col(success_col) == 1, 1).otherwise(0)).alias("successes")
        ).collect()[0]
        
        control_total = int(control_metrics.total) if control_metrics.total else 0
        control_success = int(control_metrics.successes) if control_metrics.successes else 0
        control_rate = float(control_success) / float(control_total) if control_total > 0 else 0
        
        # Calculate lifts
        absolute_lift = action_rate - control_rate
        relative_lift = (action_rate / control_rate - 1) if control_rate > 0 else 0
        
        # Statistical significance
        sig_test = calculate_statistical_significance(
            action_success, action_total,
            control_success, control_total
        )
        
        # Highlight top performers
        row_style = ""
        if action_rate > 0.5:  # >50% success rate
            row_style = " style='background-color: #ccffcc;'"
        elif action_rate < 0.01:  # <1% success rate
            row_style = " style='background-color: #ffcccc;'"
        
        html_table += f"""
        <tr{row_style}>
            <td style='padding: 8px;'>{campaign}</td>
            <td style='padding: 8px; text-align: right;'>{action_rate*100:.2f}%</td>
            <td style='padding: 8px; text-align: right;'>{control_rate*100:.2f}%</td>
            <td style='padding: 8px; text-align: right;'>{absolute_lift*100:.2f}%</td>
            <td style='padding: 8px; text-align: right;'>{relative_lift*100:.1f}%</td>
            <td style='padding: 8px; text-align: right;'>{sig_test['p_value']:.4f}</td>
            <td style='padding: 8px;'>{'Yes ✓' if sig_test['is_significant'] else 'No'}</td>
        </tr>
        """
    
    html_table += "</table>"
    display(HTML(html_table))
    
    # ========================================================================
    # SECTION 6: KEY INSIGHTS AND RECOMMENDATIONS
    # ========================================================================
    display(HTML("<h2>6. KEY INSIGHTS FROM CAMPAIGN ANALYSIS</h2>"))
    
    insights = []
    
    # Insight 1: Campaign volume imbalance
    if campaign_metrics:
        # Find campaign with most deployments
        max_campaign = None
        max_deployments = 0
        total_deployments = 0
        
        for m in campaign_metrics:
            total_deployments += m['action_deployments']
            if m['action_deployments'] > max_deployments:
                max_deployments = m['action_deployments']
                max_campaign = m['campaign']
        
        if max_campaign and total_deployments > 0:
            dominance = float(max_deployments) / float(total_deployments) * 100
            if dominance > 50:
                insights.append(f"Campaign {max_campaign} dominates with {dominance:.1f}% of all deployments")
    
    # Insight 2: Over-contacting within campaigns
    for m in campaign_metrics:
        if m['max_contacts'] > 10:
            insights.append(f"Campaign {m['campaign']} shows over-contacting with up to {m['max_contacts']} contacts per client")
    
    # Insight 3: Overlap issues
    if total_overlap_clients > 0:
        overlap_pct = float(total_overlap_clients) / float(df_action.select("CLNT_NO").distinct().count()) * 100
        insights.append(f"{overlap_pct:.1f}% of clients experience campaign overlaps within 25-day windows")
    
    # Display insights
    if insights:
        html_list = "<ul>"
        for insight in insights:
            html_list += f"<li>{insight}</li>"
        html_list += "</ul>"
        display(HTML(html_list))
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    display(HTML("<hr>"))
    display(HTML("<p><b>Phase 2 Analysis Complete</b> - Campaign interaction patterns analyzed</p>"))
    display(HTML("<p><i>Ready for Phase 3: Journey Analysis</i></p>"))


# ========================================================================
# MAIN EXECUTION BLOCK
# ========================================================================
if __name__ == "__main__":
    """
    To run this analysis:
    1. Ensure Spark session is initialized
    2. Load the data from HDFS
    3. Call generate_phase2_analysis(df)
    
    Example:
    --------
    spark = SparkSession.builder \
        .appName("VVD_Campaign_Analysis_Phase2") \
        .enableHiveSupport() \
        .getOrCreate()
    
    df = spark.read.parquet("/user/427966379/final_df.parquet")
    generate_phase2_analysis(df)
    """
    pass

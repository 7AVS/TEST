"""
================================================================================
PHASE 1: UNIVERSE OVERVIEW (40,000 FEET VIEW) - VERSION 2
================================================================================
Purpose: Establish baseline metrics for VVD campaign analysis
Output: HTML tables displayed directly in Jupyter notebook cells
Author: Campaign Analysis Team
Date: July 2025
Version: 2.0

CHANGES IN V2:
- Fixed PySpark Row object handling (no Python built-in conflicts)
- Fixed success rate calculations showing zeros
- Added statistical significance testing
- Added Sample Ratio Mismatch (SRM) detection
- Enhanced success metrics with volume details

IMPORTANT: This is an exploratory analysis. No external files are created.
           The executed notebook serves as the audit trail.
           
Data Source: /user/427966379/final_df.parquet
================================================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from IPython.display import display, HTML
from datetime import datetime
import math


def calculate_statistical_significance(success_a, total_a, success_b, total_b, alpha=0.05):
    """
    Calculate statistical significance using z-test for proportions
    
    Parameters:
    -----------
    success_a : int - Number of successes in group A (action)
    total_a : int - Total observations in group A
    success_b : int - Number of successes in group B (control)
    total_b : int - Total observations in group B
    alpha : float - Significance level (default 0.05)
    
    Returns:
    --------
    dict with p_value, is_significant, and confidence_interval
    """
    # Avoid division by zero
    if total_a == 0 or total_b == 0:
        return {
            'p_value': 1.0,
            'is_significant': False,
            'confidence_interval': (0, 0)
        }
    
    # Calculate proportions
    p_a = success_a / total_a
    p_b = success_b / total_b
    
    # Pooled proportion
    p_pooled = (success_a + success_b) / (total_a + total_b)
    
    # Standard error
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/total_a + 1/total_b))
    
    # Z-score
    if se == 0:
        z_score = 0
    else:
        z_score = (p_a - p_b) / se
    
    # P-value (two-tailed test)
    # Using approximation for normal distribution
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


def calculate_sample_ratio_mismatch(actual_a, actual_b, expected_ratio=1.0):
    """
    Calculate Sample Ratio Mismatch (SRM) to detect randomization issues
    
    Parameters:
    -----------
    actual_a : int - Actual count in group A
    actual_b : int - Actual count in group B
    expected_ratio : float - Expected ratio of A/B (default 1.0 for 50/50 split)
    
    Returns:
    --------
    dict with srm_detected, chi_square, and p_value
    """
    total = actual_a + actual_b
    if total == 0:
        return {
            'srm_detected': False,
            'chi_square': 0,
            'p_value': 1.0
        }
    
    # Expected counts based on ratio
    expected_a = total * (expected_ratio / (1 + expected_ratio))
    expected_b = total * (1 / (1 + expected_ratio))
    
    # Chi-square test
    chi_square = ((actual_a - expected_a) ** 2 / expected_a + 
                  (actual_b - expected_b) ** 2 / expected_b)
    
    # P-value approximation for chi-square with 1 degree of freedom
    # Using approximation: p ≈ 1 - Φ(√χ²)
    p_value = 1 - 0.5 * (1 + math.erf(math.sqrt(chi_square) / math.sqrt(2)))
    
    return {
        'srm_detected': p_value < 0.05,
        'chi_square': chi_square,
        'p_value': p_value,
        'expected_a': expected_a,
        'expected_b': expected_b
    }


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
    display(HTML("<h1>PHASE 1: UNIVERSE OVERVIEW - BASELINE METRICS (v2)</h1>"))
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
    df_control = df.filter(F.col("TST_GRP_CD") == "TG7")
    
    
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
    avg_deployments_per_client = float(total_deployments) / float(unique_clients) if unique_clients > 0 else 0
    
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
    # Using explicit loop to avoid built-in conflicts
    total_all_deployments = 0
    for row in test_group_split:
        total_all_deployments += int(row.deployments)
    
    # Check for Sample Ratio Mismatch
    action_count = 0
    control_count = 0
    for row in test_group_split:
        if row.TST_GRP_CD == "TG4":
            action_count = int(row.deployments)
        elif row.TST_GRP_CD == "TG7":
            control_count = int(row.deployments)
    
    srm_result = calculate_sample_ratio_mismatch(action_count, control_count)
    
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
        pct = float(row.deployments) / float(total_all_deployments) * 100 if total_all_deployments > 0 else 0
        
        # Highlight the test group codes for clarity
        group_desc = "Action Group" if row.TST_GRP_CD == "TG4" else "Control Group"
        
        html_table += f"""
        <tr>
            <td style='padding: 8px;'>{row.TST_GRP_CD} ({group_desc})</td>
            <td style='padding: 8px; text-align: right;'>{int(row.deployments):,}</td>
            <td style='padding: 8px; text-align: right;'>{int(row.unique_clients):,}</td>
            <td style='padding: 8px; text-align: right;'>{pct:.1f}%</td>
        </tr>
        """
    
    html_table += "</table>"
    display(HTML(html_table))
    
    # Display SRM results
    if srm_result['srm_detected']:
        display(HTML(f"<p style='color: red;'><b>⚠️ Sample Ratio Mismatch Detected!</b> p-value: {srm_result['p_value']:.4f}</p>"))
    else:
        display(HTML(f"<p style='color: green;'>✓ No Sample Ratio Mismatch detected (p-value: {srm_result['p_value']:.4f})</p>"))
    
    
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
        pct = float(row.deployments) / float(total_deployments) * 100 if total_deployments > 0 else 0
        
        # Calculate average contacts per client for this campaign
        avg_per_client = float(row.deployments) / float(row.unique_clients) if row.unique_clients > 0 else 0
        
        # Highlight if a campaign is dominating (>50% of deployments)
        row_style = " style='background-color: #ffe6e6;'" if pct > 50 else ""
        
        html_table += f"""
        <tr{row_style}>
            <td style='padding: 8px;'>{row.MNE}</td>
            <td style='padding: 8px; text-align: right;'>{int(row.deployments):,}</td>
            <td style='padding: 8px; text-align: right;'>{int(row.unique_clients):,}</td>
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
    freq_dict = {}
    for row in freq_buckets:
        freq_dict[row.frequency_bucket] = int(row.count)
    
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
            pct = float(count) / float(unique_clients) * 100 if unique_clients > 0 else 0
            
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
    monthly_rate = 0  # Initialize to avoid scope issues
    if months_covered > 0:
        monthly_rate = float(total_deployments) / (float(unique_clients) * float(months_covered))
        
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
    # SECTION 5: SUCCESS RATE BASELINES (ENHANCED)
    # ========================================================================
    # This section establishes baseline success rates for lift calculations.
    # We compare action group (TG4) vs control group (TG7).
    # ENHANCED: Shows volumes and statistical significance
    
    display(HTML("<h2>5. SUCCESS RATE BASELINES (Enhanced with Statistical Testing)</h2>"))
    display(HTML("<p>Baseline success rates with volume details and statistical significance</p>"))
    
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
    
    # First, let's check which success columns actually exist in the data
    df_columns = df.columns
    existing_success_cols = []
    
    # Check each possible success column
    if 'acquisition_success' in df_columns:
        existing_success_cols.append('acquisition_success')
    if 'activation_success' in df_columns:
        existing_success_cols.append('activation_success')
    if 'usage_success' in df_columns:
        existing_success_cols.append('usage_success')
    if 'provisioning_success' in df_columns:
        existing_success_cols.append('provisioning_success')
    
    display(HTML(f"<p><b>Available success columns:</b> {', '.join(existing_success_cols) if existing_success_cols else 'None found'}</p>"))
    
    # Overall Success Metrics with Volume Details
    display(HTML("<h3>Overall Success Metrics</h3>"))
    
    html_table = """
    <table border='1' style='border-collapse: collapse;'>
    <tr style='background-color: #f2f2f2;'>
        <th style='padding: 8px;'>Metric</th>
        <th style='padding: 8px; text-align: right;'>Action (TG4)</th>
        <th style='padding: 8px; text-align: right;'>Control (TG7)</th>
        <th style='padding: 8px; text-align: right;'>Lift</th>
        <th style='padding: 8px; text-align: right;'>P-Value</th>
        <th style='padding: 8px;'>Significant?</th>
    </tr>
    """
    
    # Calculate deployments for each group
    action_deployments = df_action.count()
    control_deployments = df_control.count()
    
    # Calculate unique clients for each group
    action_unique_clients = df_action.select("CLNT_NO").distinct().count()
    control_unique_clients = df_control.select("CLNT_NO").distinct().count()
    
    # Add deployment and client rows
    html_table += f"""
    <tr>
        <td style='padding: 8px;'>Total Deployments</td>
        <td style='padding: 8px; text-align: right;'>{action_deployments:,}</td>
        <td style='padding: 8px; text-align: right;'>{control_deployments:,}</td>
        <td style='padding: 8px; text-align: right;'>-</td>
        <td style='padding: 8px; text-align: right;'>-</td>
        <td style='padding: 8px;'>-</td>
    </tr>
    <tr>
        <td style='padding: 8px;'>Unique Clients</td>
        <td style='padding: 8px; text-align: right;'>{action_unique_clients:,}</td>
        <td style='padding: 8px; text-align: right;'>{control_unique_clients:,}</td>
        <td style='padding: 8px; text-align: right;'>-</td>
        <td style='padding: 8px; text-align: right;'>-</td>
        <td style='padding: 8px;'>-</td>
    </tr>
    """
    
    # Calculate overall success if we have success columns
    if existing_success_cols:
        # Create a column that checks if ANY success occurred
        success_expr_list = [F.col(col) for col in existing_success_cols]
        
        # Calculate successes for action group
        action_success_df = df_action.select(
            F.greatest(*success_expr_list).alias("any_success")
        ).agg(
            F.sum(F.when(F.col("any_success") == 1, 1).otherwise(0)).alias("success_count"),
            F.count("*").alias("total_count")
        ).collect()[0]
        
        action_success_count = int(action_success_df.success_count) if action_success_df.success_count else 0
        action_total_count = int(action_success_df.total_count)
        action_success_rate = float(action_success_count) / float(action_total_count) if action_total_count > 0 else 0
        
        # Calculate successes for control group
        control_success_df = df_control.select(
            F.greatest(*success_expr_list).alias("any_success")
        ).agg(
            F.sum(F.when(F.col("any_success") == 1, 1).otherwise(0)).alias("success_count"),
            F.count("*").alias("total_count")
        ).collect()[0]
        
        control_success_count = int(control_success_df.success_count) if control_success_df.success_count else 0
        control_total_count = int(control_success_df.total_count)
        control_success_rate = float(control_success_count) / float(control_total_count) if control_total_count > 0 else 0
        
        # Calculate lift and statistical significance
        lift = action_success_rate - control_success_rate
        sig_test = calculate_statistical_significance(
            action_success_count, action_total_count,
            control_success_count, control_total_count
        )
        
        # Add success metrics rows
        html_table += f"""
        <tr>
            <td style='padding: 8px;'>Successful Deployments</td>
            <td style='padding: 8px; text-align: right;'>{action_success_count:,}</td>
            <td style='padding: 8px; text-align: right;'>{control_success_count:,}</td>
            <td style='padding: 8px; text-align: right;'>-</td>
            <td style='padding: 8px; text-align: right;'>-</td>
            <td style='padding: 8px;'>-</td>
        </tr>
        <tr>
            <td style='padding: 8px;'><b>Success Rate</b></td>
            <td style='padding: 8px; text-align: right;'><b>{action_success_rate*100:.2f}%</b></td>
            <td style='padding: 8px; text-align: right;'><b>{control_success_rate*100:.2f}%</b></td>
            <td style='padding: 8px; text-align: right;'><b>{lift*100:.2f}%</b></td>
            <td style='padding: 8px; text-align: right;'><b>{sig_test['p_value']:.4f}</b></td>
            <td style='padding: 8px;'><b>{'Yes ✓' if sig_test['is_significant'] else 'No'}</b></td>
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
        <th style='padding: 8px; text-align: right;'>Action Deployments</th>
        <th style='padding: 8px; text-align: right;'>Control Deployments</th>
        <th style='padding: 8px; text-align: right;'>Action Rate</th>
        <th style='padding: 8px; text-align: right;'>Control Rate</th>
        <th style='padding: 8px; text-align: right;'>Lift</th>
        <th style='padding: 8px; text-align: right;'>P-Value</th>
        <th style='padding: 8px;'>Significant?</th>
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
        if success_col in df_columns:
            # Get deployment counts and success metrics for action group
            action_metrics = df_action.filter(F.col("MNE").isin(campaigns)).agg(
                F.count("*").alias("total_deployments"),
                F.sum(F.when(F.col(success_col) == 1, 1).otherwise(0)).alias("success_count")
            ).collect()[0]
            
            action_total = int(action_metrics.total_deployments) if action_metrics.total_deployments else 0
            action_success = int(action_metrics.success_count) if action_metrics.success_count else 0
            action_rate = float(action_success) / float(action_total) if action_total > 0 else 0
            
            # Get deployment counts and success metrics for control group
            control_metrics = df_control.filter(F.col("MNE").isin(campaigns)).agg(
                F.count("*").alias("total_deployments"),
                F.sum(F.when(F.col(success_col) == 1, 1).otherwise(0)).alias("success_count")
            ).collect()[0]
            
            control_total = int(control_metrics.total_deployments) if control_metrics.total_deployments else 0
            control_success = int(control_metrics.success_count) if control_metrics.success_count else 0
            control_rate = float(control_success) / float(control_total) if control_total > 0 else 0
            
            # Calculate lift and significance
            lift = action_rate - control_rate
            sig_test = calculate_statistical_significance(
                action_success, action_total,
                control_success, control_total
            )
            
            html_table += f"""
            <tr>
                <td style='padding: 8px;'>{campaign_type}</td>
                <td style='padding: 8px; text-align: right;'>{action_total:,}</td>
                <td style='padding: 8px; text-align: right;'>{control_total:,}</td>
                <td style='padding: 8px; text-align: right;'>{action_rate*100:.2f}%</td>
                <td style='padding: 8px; text-align: right;'>{control_rate*100:.2f}%</td>
                <td style='padding: 8px; text-align: right;'>{lift*100:.2f}%</td>
                <td style='padding: 8px; text-align: right;'>{sig_test['p_value']:.4f}</td>
                <td style='padding: 8px;'>{'Yes ✓' if sig_test['is_significant'] else 'No'}</td>
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
        dominance_pct = float(top_campaign.deployments) / float(total_deployments) * 100
        if dominance_pct > 50:
            observations.append(f"{top_campaign.MNE} dominates with {dominance_pct:.1f}% of all deployments")
    
    # Observation 2: Check for over-contacting
    if "11+ contacts" in freq_dict:
        over_contacted_pct = float(freq_dict["11+ contacts"]) / float(unique_clients) * 100
        if over_contacted_pct > 20:
            observations.append(f"{over_contacted_pct:.1f}% of clients receive 11+ contacts (potential over-contacting)")
    
    # Observation 3: Monthly contact rate insight
    if months_covered > 0 and monthly_rate > 0:
        if monthly_rate > 1:
            observations.append(f"Average client receives {monthly_rate:.1f} contacts per month")
    
    # Observation 4: Test/Control split insight
    action_group = None
    for row in test_group_split:
        if row.TST_GRP_CD == "TG4":
            action_group = row
            break
    
    if action_group and total_all_deployments > 0:
        action_pct = float(action_group.deployments) / float(total_all_deployments) * 100
        observations.append(f"Action group (TG4) represents {action_pct:.1f}% of total deployments")
    
    # Observation 5: SRM warning if detected
    if srm_result['srm_detected']:
        observations.append("⚠️ Sample Ratio Mismatch detected - randomization may be compromised")
    
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
    display(HTML("<p><i>Version 2.0 - Enhanced with statistical testing and volume details</i></p>"))


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
        .appName("VVD_Campaign_Analysis_Phase1_v2") \
        .enableHiveSupport() \
        .getOrCreate()
    
    df = spark.read.parquet("/user/427966379/final_df.parquet")
    generate_phase1_analysis(df)
    """
    
    # Note: The actual execution will be done in the Jupyter notebook
    # This block is here for documentation purposes
    pass

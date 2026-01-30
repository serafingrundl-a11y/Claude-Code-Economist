"""
Create figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = r"C:\Users\seraf\DACA Results Task 1\replication_10"

def create_event_study_plot():
    """Create event study figure showing parallel trends and treatment effects."""

    # Event study coefficients
    years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
    coefs = [-0.05039, -0.03662, -0.02128, -0.00399, 0.00498, 0, 0.02583, 0.03860, 0.06259, 0.07203]
    ses = [0.00993, 0.00968, 0.00982, 0.00961, 0.00940, 0, 0.00939, 0.00946, 0.00938, 0.00961]

    ci_lower = [c - 1.96*s for c, s in zip(coefs, ses)]
    ci_upper = [c + 1.96*s for c, s in zip(coefs, ses)]

    plt.figure(figsize=(10, 6))

    # Plot confidence intervals
    plt.fill_between(years, ci_lower, ci_upper, alpha=0.3, color='steelblue')

    # Plot point estimates
    plt.plot(years, coefs, 'o-', color='steelblue', linewidth=2, markersize=8)

    # Add horizontal line at zero
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    # Add vertical line at treatment (between 2011 and 2012, so we put at 2012)
    plt.axvline(x=2012, color='red', linestyle='--', linewidth=1, label='DACA Implementation (2012)')

    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Coefficient (Relative to 2011)', fontsize=12)
    plt.title('Event Study: DACA Effect on Full-Time Employment', fontsize=14)
    plt.legend(loc='upper left')
    plt.xticks(years)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'event_study_plot.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'event_study_plot.pdf'), bbox_inches='tight')
    plt.close()
    print("Saved event_study_plot.png and .pdf")


def create_trends_plot():
    """Create pre/post trends plot for treatment and control groups."""

    # Data from analysis output (weighted means)
    years_pre = [2006, 2007, 2008, 2009, 2010, 2011]
    years_post = [2013, 2014, 2015, 2016]

    # Approximate full-time employment rates by year (will need to compute these)
    # Using the summary stats: Pre eligible: 0.3991, Post eligible: 0.4799
    # Pre ineligible: 0.5754, Post ineligible: 0.5683

    # For now, create a simpler version showing pre/post means
    groups = ['DACA Eligible', 'DACA Ineligible']
    pre_means = [0.3991, 0.5754]
    post_means = [0.4799, 0.5683]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    bars1 = ax.bar(x - width/2, pre_means, width, label='Pre-DACA (2006-2011)', color='steelblue')
    bars2 = ax.bar(x + width/2, post_means, width, label='Post-DACA (2013-2016)', color='coral')

    ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
    ax.set_title('Full-Time Employment by DACA Eligibility and Period', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.set_ylim(0, 0.7)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'did_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'did_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("Saved did_comparison.png and .pdf")


def create_robustness_forest_plot():
    """Create forest plot of robustness check results."""

    # Results from analysis
    specifications = [
        'Main Specification',
        'Age 18-35 Only',
        'Any Employment',
        'Males Only',
        'Females Only',
        'Unweighted'
    ]

    coefficients = [0.06537, 0.03540, 0.08425, 0.06316, 0.05920, 0.06986]
    std_errors = [0.00433, 0.00512, 0.00423, 0.00590, 0.00615, 0.00356]

    ci_lower = [c - 1.96*s for c, s in zip(coefficients, std_errors)]
    ci_upper = [c + 1.96*s for c, s in zip(coefficients, std_errors)]

    y_positions = range(len(specifications))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot point estimates and confidence intervals
    for i, (coef, lower, upper) in enumerate(zip(coefficients, ci_lower, ci_upper)):
        color = 'darkblue' if i == 0 else 'steelblue'
        ax.plot([lower, upper], [i, i], color=color, linewidth=2)
        ax.plot(coef, i, 'o', color=color, markersize=10)

    # Add vertical line at zero
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(specifications)
    ax.set_xlabel('Coefficient (Effect on Full-Time Employment)', fontsize=12)
    ax.set_title('Robustness Checks: DACA Effect Estimates', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'robustness_forest.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'robustness_forest.pdf'), bbox_inches='tight')
    plt.close()
    print("Saved robustness_forest.png and .pdf")


def main():
    print("Creating figures...")
    create_event_study_plot()
    create_trends_plot()
    create_robustness_forest_plot()
    print("All figures created successfully!")


if __name__ == "__main__":
    main()

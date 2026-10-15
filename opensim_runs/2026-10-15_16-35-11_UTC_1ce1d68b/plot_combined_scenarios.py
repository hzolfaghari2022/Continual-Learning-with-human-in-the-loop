"""Render constant-load + changing-load traces from computed CSVs.

Used automatically by compare_fixed_PI_loads.py. Can also replot a completed
comparison folder without simulating: python plot_combined_scenarios.py RESULTS.
"""
import csv
import json
from pathlib import Path
import math

def read_csv(path):
    with Path(path).open(newline='') as f:
        return [{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]

def plot_combined(plt,cases,kp,ki,capacity,reference_times,refs,schedule,dest):
    """cases: (label, rows, changing, completed). Does not edit or invent data."""
    colors=['#0077B6','#159276','#D76523','#5975ab','#9e6c20']
    fig,axes=plt.subplots(4,1,figsize=(13,11),sharex=True,
                         gridspec_kw={'height_ratios':[.75,1.3,1,1]})
    fig.subplots_adjust(top=.86,bottom=.1,left=.09,right=.98,hspace=.22)
    constant=0
    for label,rows,changing,completed in cases:
        if not rows:continue
        if any(abs(r['Kp']-kp)>1e-10 or abs(r['Ki']-ki)>1e-10 for r in rows):
            raise ValueError('Combined fixed-PI figure contains a different gain.')
        dt=reference_times[1]-reference_times[0]
        for row in rows:
            k=round((row['time_s']-reference_times[0])/dt)
            if k<0 or k>=len(refs) or abs(row['time_s']-reference_times[k])>1e-7 or abs(row['reference_deg']-refs[k])>1e-7:
                raise ValueError('Scenario target/time grid differs from the common reference.')
            if row['tension_N']>capacity+1e-6:
                raise ValueError('Scenario force limit differs from the combined plot label.')
        color='#7C3BA0' if changing else colors[constant%len(colors)]
        if not changing:constant+=1
        t=[r['time_s'] for r in rows]
        label+=(' (FAILED: partial)' if not completed else '')
        axes[0].step(t,[r['load_kg'] for r in rows],where='post',color=color,
                     lw=2.4 if changing else 1.4,ls='-' if changing else '--')
        for ax,key in zip(axes[1:],['angle_deg','error_deg','tension_N']):
            ax.plot(t,[r[key] for r in rows],color=color,lw=1.9 if changing else 1.3,label=label)
    axes[1].plot(reference_times,refs,'k--',lw=1.5,label='Target',zorder=5)
    axes[0].set_ylabel('Dumbbell\nmass (kg)')
    axes[1].set_ylabel('Elbow angle\n(deg)')
    axes[2].set_ylabel('Target − actual\n(deg)');axes[2].axhline(0,color='gray',lw=.8)
    axes[3].set_ylabel('Cable tension\n(N)');axes[3].axhline(capacity,color='#333333',ls=':',lw=1.7)
    axes[-1].set_xlabel('Simulation time (s)')
    duration=reference_times[-1]
    for ax in axes:
        ax.grid(alpha=.18);ax.set_xlim(0,duration)
        for task in schedule[1:]:ax.axvline(task['time_s'],color='#7C3BA0',ls=':',alpha=.45,lw=1)
    mass_max=max([r['load_kg'] for _,rows,_,_ in cases for r in rows],default=1)
    axes[0].set_ylim(0,mass_max*1.15+.1)
    changed=next(((rows,ok) for _,rows,is_changing,ok in cases if is_changing),([],False))
    observed_end=changed[0][-1]['time_s'] if changed[0] else 0
    for i,task in enumerate(schedule):
        if task['time_s']>observed_end:continue
        end=min(schedule[i+1]['time_s'] if i+1<len(schedule) else duration,observed_end)
        axes[0].text((task['time_s']+end)/2,mass_max*1.06,f"{task['load_kg']:g} kg",ha='center',color='#7C3BA0',fontsize=10,weight='bold')
    if not changed[1]:
        axes[0].text(.99,.04,f'Changing-load trial stopped at {observed_end:.2f} s',
                     transform=axes[0].transAxes,ha='right',fontsize=9,color='#7C3BA0')
    fig.suptitle('Arm26: all load scenarios with the SAME fixed PI controller',fontsize=18,weight='bold',y=.99)
    fig.text(.5,.955,f'Kp = {kp:g} N·m/rad | Ki = {ki:g} N·m/(rad·s) | Same target | Cable limit = {capacity:g} N',ha='center',fontsize=11)
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.937),ncol=3,frameon=False,fontsize=10)
    fig.text(.5,.035,'Purple dotted lines mark load changes. Constant-load cases are separate independent trials.',ha='center',fontsize=10,color='#46515b')
    dest=Path(dest)
    fig.savefig(dest/'all_scenarios_fixed_PI.png',dpi=180)
    fig.savefig(dest/'all_scenarios_fixed_PI.pdf')
    return fig

def replot(folder):
    import matplotlib.pyplot as plt
    folder=Path(folder).resolve();settings=json.loads((folder/'run_settings.json').read_text())
    capacity=settings['tension_caps_N'][0];cases=[]
    for mass in settings['loads_kg']:
        case=folder/f'load_{mass:g}kg_cap_{capacity:g}N'
        file=case/'fixed_PI_data.csv';completed=file.exists()
        if not completed:file=case/'partial_data.csv'
        rows=read_csv(file) if file.exists() else []
        cases.append((f'Constant {mass:g} kg',rows,False,completed))
    case=folder/'changing_load';file=case/'fixed_PI_data.csv';completed=file.exists()
    if not completed:file=case/'partial_data.csv'
    rows=read_csv(file) if file.exists() else []
    cases.append(('Changing load',rows,True,completed))
    schedule=settings['changing_load_schedule']
    times=[k*settings['dt_s'] for k in range(round(settings['duration_s']/settings['dt_s'])+1)]
    import run_arm26_pi_github as base
    base.DEMO_START_DEG=settings['start_deg'];base.DEMO_TARGET_DEG=settings['target_deg'];base.DEMO_CYCLES=settings['cycles']
    refs=[math.degrees(base.reference(t,settings['leg_time_s'])) for t in times]
    plot_combined(plt,cases,settings['kp'],settings['ki'],capacity,times,refs,schedule,folder)
    print('Saved:',folder/'all_scenarios_fixed_PI.png');plt.show()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path)
    replot(p.parse_args().results)

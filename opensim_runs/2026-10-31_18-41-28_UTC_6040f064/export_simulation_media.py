"""Save diagnostics and a mesh-based 3D projected replay of recorded OpenSim states.

No screen capture, synthetic trajectories, VTK, or rerun of the controller.
Uses the actual source model meshes/transforms and recorded state table.
MP4 requires ffmpeg on PATH or optional `pip install imageio-ffmpeg`.
Without ffmpeg a portable Motion-JPEG AVI is written using only Pillow/stdlib.
"""
import argparse,csv,json,math,os,shutil,struct,subprocess,sys,traceback
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parent

def windows_extended_path(path):
    """Return an absolute Windows file path that supports nested long names.

    Pure string conversion is separate from filesystem access for portable tests.
    Accept drive paths, UNC shares, and already extended paths without doubling
    their prefix. Relative names must be made absolute by filesystem_path first.
    """
    import ntpath
    value=ntpath.normpath(os.fspath(path))
    if value.startswith('\\\\?\\'):return value
    if value.startswith('\\\\.\\'):
        raise ValueError('Device paths are not supported for report exports')
    if value.startswith('\\\\'):
        return '\\\\?\\UNC\\'+value[2:]
    drive,tail=ntpath.splitdrive(value)
    if not drive or not tail.startswith('\\'):
        raise ValueError('An absolute Windows path is required')
    return '\\\\?\\'+value


def filesystem_path(path):
    """Use Unicode extended paths for Python report I/O on Windows only.

    No registry changes and no changes to OpenSim/FFmpeg input paths are needed.
    """
    value=Path(path).expanduser().resolve()
    return Path(windows_extended_path(value)) if os.name=='nt' else value

def load_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:return [{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]

def read_states(path):
    lines=Path(path).read_text().splitlines();i=next(i for i,x in enumerate(lines) if x.strip()=='endheader')
    names=lines[i+1].split();a=np.loadtxt(lines[i+2:]);return names,a

def diagnostic_figures(folder):
    import matplotlib
    # Export is intended to be independent of the interactive viewer.
    import matplotlib.pyplot as plt
    folder=Path(folder);a=load_csv(folder/'stable_penalty_data.csv');t=np.array([r['time_s'] for r in a]);arr=lambda k:np.array([r[k] for r in a])
    settings=json.loads((folder/'run_settings.json').read_text())
    def save(fig,name):
        fig.savefig(folder/(name+'.png'),dpi=200,bbox_inches='tight');fig.savefig(folder/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(11,8),layout='constrained')
    ax=axs.flat
    ax[0].plot(t,arr('reference_deg'),'k--',label='Reference');ax[0].plot(t,arr('angle_deg'),label='Actual',color='#007d9d');ax[0].set_ylabel('Angle (deg)');ax[0].legend(fontsize=8)
    ax[1].plot(t,arr('error_deg'),color='#d65e32');ax[1].set_ylabel('Tracking error (deg)')
    ax[2].plot(t,arr('tension_N'),color='#00885d');ax[2].axhline(settings['tmax'],color='gray',ls=':');ax[2].set_ylabel('Cable tension (N)')
    for k,label in [('P_Nm','P'),('I_Nm','I'),('handcoded_PD_Nm','Fixed PD'),('learned_only_Nm','Learned'),('torque_applied_Nm','Applied')]:ax[3].plot(t,arr(k),label=label,lw=1)
    ax[3].set_ylabel('Torque (N m)');ax[3].legend(fontsize=7,ncol=3)
    for muscle,label in [('BIClong','Biceps long'),('TRIlong','Triceps long')]:
        ax[4].plot(t,arr(muscle+'_fiber_length_m'),label=label);ax[5].plot(t,arr(muscle+'_tendon_force_N'),label=label)
    ax[4].set_ylabel('Fiber length (m)');ax[5].set_ylabel('Tendon force (N)')
    for x in ax:
        for event in settings['load_schedule'][1:]:x.axvline(event['time_s'],color='gray',ls=':',lw=.7)
        x.set_xlabel('Time (s)');x.set_xlim(t[0],t[-1]);x.grid(alpha=.2)
    ax[4].legend(fontsize=8);ax[5].legend(fontsize=8);fig.suptitle('Updated controller: saved dashboard from recorded simulation data');save(fig,'simulation_dashboard')
    e=np.radians(arr('error_deg'));vref=np.gradient(np.radians(arr('reference_deg')),t)
    # Use the exact forward difference reference velocity used by the runner.
    import arm26_ewc_original_core as core
    core.base.DEMO_START_DEG=settings['start_deg'];core.base.DEMO_TARGET_DEG=settings['target_deg'];core.base.DEMO_CYCLES=settings['cycles']
    vref=np.array([core.ref_and_vel(float(x),settings['rise_time'])[1] for x in t])
    s=settings['kp']*e+settings['kd_fel']*(vref-arr('velocity_rad_s'))
    track=.5*e*e;task=-s*arr('learned_only_Nm');penalty=arr('ewc_penalty');penalty[-1]=np.nan
    fig,axs=plt.subplots(3,1,figsize=(10,6.2),sharex=True,layout='constrained')
    for x,y,c,label in zip(axs,[track,task,penalty],['#007d9d','#d65e32','#785da0'],['Tracking diagnostic: 0.5 e^2 (rad^2)','Local FEL surrogate: -s h(w)','Memory penalty R(w)']):
        x.plot(t,y,color=c,lw=1);x.set_ylabel(label,fontsize=9);x.grid(alpha=.2)
        for event in settings['load_schedule'][1:]:x.axvline(event['time_s'],color='gray',ls=':',lw=.7)
    axs[1].axhline(0,color='gray',lw=.6);axs[-1].set_xlabel('Time (s)');fig.suptitle('Learning diagnostics: local surrogate and tracking loss are different quantities');save(fig,'learning_diagnostics')
    with (folder/'learning_diagnostics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['time_s','tracking_half_squared_rad','sliding_teacher_s','local_FEL_surrogate','memory_penalty']);w.writerows(zip(t,track,s,task,penalty))
    p=folder/'stable_penalty/task_checkpoints.npz'
    if p.is_file():
        d=np.load(p);fig,axs=plt.subplots(1,2,figsize=(10,3.1),layout='constrained')
        im=axs[0].imshow(d['importance_proxy'],aspect='auto',cmap='viridis');fig.colorbar(im,ax=axs[0],label='Sensitivity importance')
        axs[0].set_yticks(range(len(d['masses'])),[f"After segment {int(s)} ({m:g} kg)" for s,m in zip(d['segments'],d['masses'])],fontsize=8);axs[0].set_xlabel('Adaptive coefficient index (0–31)')
        axs[1].plot(d['segments'],np.linalg.norm(d['weights'],axis=1),'o-',color='#007d9d');axs[1].set(xlabel='Completed segment',ylabel='Norm of learned coefficient vector');axs[1].grid(alpha=.2);axs[1].set_xticks(d['segments'])
        fig.suptitle('Stored memory diagnostics; neither panel by itself proves retention');save(fig,'memory_diagnostics')
    return ['simulation_dashboard','learning_diagnostics','memory_diagnostics']


def vtp_triangles(path):
    root=ET.parse(path);points=root.find('.//Points/DataArray')
    if points.get('format')!='ascii':raise ValueError('This portable renderer requires ASCII VTP meshes: '+str(path))
    p=np.fromstring(points.text,sep=' ').reshape(-1,3);arrays=root.findall('.//Polys/DataArray')
    conn=next(x for x in arrays if x.get('Name')=='connectivity');offs=next(x for x in arrays if x.get('Name')=='offsets')
    conn=np.fromstring(conn.text,sep=' ',dtype=int);offs=np.fromstring(offs.text,sep=' ',dtype=int);tris=[];start=0
    for stop in offs:
        face=conn[start:stop];start=stop
        for j in range(1,len(face)-1):tris.append([face[0],face[j],face[j+1]])
    return p[np.array(tris)]

def simplify_for_rendering(tri,spacing=.0015):
    """Sub-pixel vertex clustering for replay only; never changes the .osim model."""
    points=tri.reshape(-1,3)
    cells=np.round(points/spacing).astype(np.int64)
    _,inverse=np.unique(cells,axis=0,return_inverse=True)
    count=np.bincount(inverse)
    vertices=np.column_stack([np.bincount(inverse,weights=points[:,i])/count for i in range(3)])
    faces=inverse.reshape(-1,3)
    faces=faces[(faces[:,0]!=faces[:,1])&(faces[:,0]!=faces[:,2])&(faces[:,1]!=faces[:,2])]
    _,first=np.unique(np.sort(faces,axis=1),axis=0,return_index=True)
    return vertices[faces[np.sort(first)]]


def cylinder(radius,height,n=20):
    theta=np.linspace(0,2*np.pi,n,endpoint=False);lo=np.c_[radius*np.cos(theta),np.full(n,-height/2),radius*np.sin(theta)];hi=lo.copy();hi[:,1]=height/2
    tris=[]
    for i in range(n):
        j=(i+1)%n;tris.extend([[lo[i],lo[j],hi[i]],[hi[i],lo[j],hi[j]],[np.array([0,-height/2,0]),lo[j],lo[i]],[np.array([0,height/2,0]),hi[i],hi[j]]])
    return np.array(tris)

def font(size):
    from PIL import ImageFont
    for candidate in ['C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(candidate).exists():return ImageFont.truetype(candidate,size)
    return ImageFont.load_default()

class ReplayRenderer:
    """Pillow rasterization of OpenSim's 3D meshes at saved states; no screen capture."""
    def __init__(self,folder,geometry=None,width=1120,height=720):
        import opensim as osim
        import run_variable_load as variable
        from types import SimpleNamespace
        from PIL import Image
        self.osim=osim;self.Image=Image;self.folder=Path(folder);self.W=width;self.H=height
        self.rows=load_csv(self.folder/'stable_penalty_data.csv');self.times=np.array([r['time_s'] for r in self.rows]);self.settings=json.loads((self.folder/'run_settings.json').read_text())
        source=self.folder/'input_model.osim';geo=Path(geometry) if geometry else ROOT/'Geometry'
        osim.ModelVisualizer.addDirToGeometrySearchPaths(str(geo.resolve()));osim.Logger.setLevelString('Error')
        args=SimpleNamespace(start_deg=self.settings['start_deg'],grip=self.settings['grip'])
        # Mass changes do not alter frames/geometry layout. Geometry is reconstructed
        # from the same build function; motion comes exclusively from saved states.
        self.model,self.state,self.elbow,self.cable=variable.build(osim,source,self.rows[0]['load_kg'],args)
        names,states=read_states(self.folder/'stable_penalty/ewc_NN_states.sto')
        actual=[self.model.getStateVariableNames().get(i) for i in range(self.model.getStateVariableNames().getSize())]
        if names[1:]!=actual:raise ValueError('Recorded state layout differs from the render model')
        self.states=states;self.entries=[]
        xml=ET.fromstring(self.model.dump());frames=[('ground',xml.find('.//Ground'))]
        for body in xml.findall('.//BodySet/objects/Body'):
            path='/bodyset/'+body.get('name');frames.append((path,body))
            for f in body.findall('./components/PhysicalOffsetFrame'):frames.append((path+'/'+f.get('name'),f))
        for path,node in frames:
            geoms=node.find('attached_geometry') if node is not None else None
            if geoms is None:continue
            frame=osim.PhysicalFrame.safeDownCast(self.model.getComponent(path))
            for geom in geoms:
                if geom.tag=='Mesh':
                    file=geo/Path(geom.findtext('mesh_file')).name
                    if not file.is_file():raise FileNotFoundError('Video mesh missing: '+str(file))
                    tri=simplify_for_rendering(vtp_triangles(file));color=np.array([202,211,218])
                elif geom.tag=='Cylinder':
                    tri=cylinder(float(geom.findtext('radius')),2*float(geom.findtext('half_height')))
                    color=np.array([29,43,57]) if 'strap' in geom.get('name','') else np.array([204,102,35])
                else:continue
                scale=np.fromstring(geom.findtext('scale_factors','1 1 1'),sep=' ');self.entries.append((frame,tri*scale,color))
        # Fixed orthographic 3D view, calibrated once for the full motion envelope.
        self.eye=np.array([1.,.12,1.55]);self.eye/=np.linalg.norm(self.eye)
        self.right=np.cross([0,1,0],self.eye);self.right/=np.linalg.norm(self.right);self.up=np.cross(self.eye,self.right)
        self.light=np.array([.3,.7,1]);self.light/=np.linalg.norm(self.light)
        self.scale=510.;self.origin=np.array([325.,365.]);self.center=np.array([0,-.20,.10]);self.ft=font(24);self.fs=font(17)
        self.trace=[(755+float(r['time_s'])/self.times[-1]*330,250-(r['reference_deg']-10)/80*150) for r in self.rows]

    def project(self,p):
        q=p-self.center;return np.stack([self.origin[0]+self.scale*(q@self.right),self.origin[1]-self.scale*(q@self.up)],axis=-1)

    def frame(self,index):
        from PIL import ImageDraw
        row=self.rows[index];vec=self.osim.Vector(len(self.states[index])-1,0.)
        for i,x in enumerate(self.states[index,1:]):vec.set(i,float(x))
        self.model.setStateVariableValues(self.state,vec);self.state.setTime(row['time_s']);self.model.realizePosition(self.state)
        triangles=[];colors=[]
        for frame,tri,color in self.entries:
            X=frame.getTransformInGround(self.state);R=np.array([[X.R().get(i,j) for j in range(3)] for i in range(3)]);p=np.array([X.p().get(i) for i in range(3)])
            world=tri@R.T+p;normal=np.cross(world[:,1]-world[:,0],world[:,2]-world[:,0]);normal/=np.maximum(1e-12,np.linalg.norm(normal,axis=1))[:,None]
            shade=.55+.45*np.abs(normal@self.light);triangles.append(world);colors.append(np.clip(color[None,:]*shade[:,None],0,255).astype('uint8'))
        tri=np.concatenate(triangles);col=np.concatenate(colors);xy=self.project(tri);depth=tri.mean(axis=1)@self.eye;order=np.argsort(depth)
        image=self.Image.new('RGB',(self.W,self.H),(242,246,250));draw=ImageDraw.Draw(image)
        for i in order:draw.polygon([tuple(x) for x in xy[i]],fill=tuple(int(v) for v in col[i]))
        # Actual muscle and cable paths, transformed by OpenSim at this state.
        muscles=self.model.getMuscles();forces=[muscles.get(i) for i in range(muscles.getSize())]+[self.cable]
        for j,force in enumerate(forces):
            path=force.getGeometryPath().getCurrentPath(self.state);pts=[]
            for i in range(path.getSize()):
                p=path.get(i);v=p.getParentFrame().findStationLocationInGround(self.state,p.getLocation(self.state));pts.append(self.project(np.array([v.get(k) for k in range(3)])))
            if len(pts)>1:draw.line([tuple(x) for x in pts],fill=(0,143,87) if j==len(forces)-1 else (191,47,49),width=4 if j==len(forces)-1 else 2)
        draw.text((24,18),'Arm26 | recorded OpenSim simulation',font=self.ft,fill=(23,50,71))
        draw.text((24,55),'Updated PI + learned torque correction',font=self.fs,fill=(80,101,119))
        draw.rounded_rectangle((725,85,1100,500),radius=12,fill='white',outline=(208,219,228),width=2)
        y=100
        for label,value in [('Time',f"{row['time_s']:05.1f} s"),('Dumbbell mass',f"{row['load_kg']:g} kg"),('Reference',f"{row['reference_deg']:.1f} deg"),('Actual angle',f"{row['angle_deg']:.1f} deg"),('Error',f"{row['error_deg']:+.2f} deg"),('Cable force',f"{row['tension_N']:.1f} N")]:
            draw.text((745,y),f'{label}: {value}',font=self.fs,fill=(23,50,71));y+=29
        draw.text((745,300),'Online learned residual',font=self.fs,fill=(0,125,157))
        draw.text((745,335),f"Correction: {row['learned_only_Nm']:.2f} N m",font=self.fs,fill=(23,50,71))
        draw.text((745,370),f"Memory penalty: {row['ewc_penalty']:.2f}",font=self.fs,fill=(23,50,71))
        draw.text((745,415),f"Regularization lambda = {self.settings['ewc_lambda']:g}",font=self.fs,fill=(80,101,119))
        draw.text((745,450),'Mass changes; visual size stays fixed.',font=self.fs,fill=(80,101,119))
        # Angle trace is placed beneath the numerical panel.
        top=550;left=745;bottom=675;right=1090
        draw.line([(left,top),(left,bottom),(right,bottom)],fill=(145,163,177),width=1)
        target=[(left+r['time_s']/self.times[-1]*(right-left),bottom-(r['reference_deg']-10)/80*(bottom-top)) for r in self.rows]
        actual=[(left+r['time_s']/self.times[-1]*(right-left),bottom-(r['angle_deg']-10)/80*(bottom-top)) for r in self.rows[:index+1]]
        draw.line(target,fill=(155,155,155),width=1)
        if len(actual)>1:draw.line(actual,fill=(0,125,157),width=2)
        cursor=left+row['time_s']/self.times[-1]*(right-left);draw.line([(cursor,top),(cursor,bottom)],fill=(212,94,50),width=1)
        draw.text((745,515),'Angle: grey target / blue actual',font=self.fs,fill=(23,50,71))
        draw.text((24,675),'Mesh-based replay of saved states; not a desktop recording.',font=self.fs,fill=(80,101,119))
        return image

class AVIWriter:
    """Minimal RIFF AVI/MJPEG fallback, for systems without FFmpeg."""
    def __init__(self,path,width,height,fps):
        self.f=Path(path).open('wb');self.entries=[];self.frames=0;self.width=width;self.height=height;self.fps=fps
        f=self.f;f.write(b'RIFF\0\0\0\0AVI ');self.hdrl=self._start_list(b'hdrl')
        self.avih_pos=f.tell()+8
        avih=struct.pack('<14I',round(1e6/fps),0,0,0x10,0,0,1,width*height*3,width,height,0,0,0,0);self._chunk(b'avih',avih)
        stream=self._start_list(b'strl');self.strh_pos=f.tell()+8
        strh=struct.pack('<4s4sIHHIIIIIIIIhhhh',b'vids',b'MJPG',0,0,0,0,1,fps,0,0,width*height*3,0xffffffff,0,0,0,width,height);self._chunk(b'strh',strh)
        strf=struct.pack('<IiiHH4sIiiII',40,width,height,1,24,b'MJPG',width*height*3,0,0,0,0);self._chunk(b'strf',strf);self._end_list(stream);self._end_list(self.hdrl)
        self.movi=self._start_list(b'movi');self.movi_data=self.movi+8
    def _chunk(self,kind,data):
        self.f.write(kind+struct.pack('<I',len(data))+data)
        if len(data)%2:self.f.write(b'\0')
    def _start_list(self,kind):
        pos=self.f.tell();self.f.write(b'LIST\0\0\0\0'+kind);return pos
    def _end_list(self,pos):
        end=self.f.tell();self.f.seek(pos+4);self.f.write(struct.pack('<I',end-pos-8));self.f.seek(end)
    def add(self,image):
        import io
        buf=io.BytesIO();image.save(buf,format='JPEG',quality=82);data=buf.getvalue();offset=self.f.tell()-self.movi_data;self.entries.append((offset,len(data)));self._chunk(b'00dc',data);self.frames+=1
    def close(self):
        self._end_list(self.movi);self._chunk(b'idx1',b''.join(struct.pack('<4sIII',b'00dc',0x10,o,s) for o,s in self.entries));end=self.f.tell()
        self.f.seek(4);self.f.write(struct.pack('<I',end-8));self.f.seek(self.avih_pos+16);self.f.write(struct.pack('<I',self.frames));self.f.seek(self.strh_pos+32);self.f.write(struct.pack('<I',self.frames));self.f.close()

def export_video(folder,geometry=None,fps=10,max_seconds=None):
    folder=Path(folder);renderer=ReplayRenderer(folder,geometry);ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:pass
    duration=renderer.times[-1];duration=min(duration,max_seconds) if max_seconds else duration
    times=np.arange(0,duration+1e-8,1/fps);indices=np.searchsorted(renderer.times,times);indices=np.minimum(indices,len(renderer.rows)-1)
    path=folder/('opensim_recorded_replay.mp4' if ffmpeg else 'opensim_recorded_replay.avi');proc=None;avi=None
    if ffmpeg:
        log=(folder/'video_encoder.log').open('w')
        proc=subprocess.Popen([ffmpeg,'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{renderer.W}x{renderer.H}','-r',str(fps),'-i','-','-an','-vcodec','libx264','-crf','24','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
    else:avi=AVIWriter(path,renderer.W,renderer.H,fps)
    stills={round(x,1):folder/f'video_frame_{i}.png' for i,x in enumerate([1.5,35.5,55.0])}
    try:
        for k,i in enumerate(indices):
            image=renderer.frame(int(i))
            if proc:proc.stdin.write(image.tobytes())
            else:avi.add(image)
            if round(float(times[k]),1) in stills:image.save(stills[round(float(times[k]),1)])
            if k%max(1,fps*10)==0:print(f'VIDEO: {times[k]:.1f}/{duration:.1f} s rendered',flush=True)
    finally:
        if proc:
            proc.stdin.close();code=proc.wait();log.close()
            if code:raise RuntimeError('Video encoding failed; see video_encoder.log')
        if avi:avi.close()
    metadata=dict(video=path.name,frames=len(indices),fps=fps,nominal_video_duration_s=len(indices)/fps,recorded_simulation_end_s=float(renderer.times[-1]),rendered_through_s=float(times[-1]),method='3D meshes transformed by saved OpenSim states; 1.5 mm visualization-only vertex clustering; no screen capture',state_source='stable_penalty/ewc_NN_states.sto',controller_source='stable_penalty_data.csv')
    (folder/'video_metadata.json').write_text(json.dumps(metadata,indent=2))
    # A compact, labelled still sequence for the editable report.
    from PIL import Image,ImageDraw
    paths=[folder/f'video_frame_{i}.png' for i in range(3)]
    if all(p.exists() for p in paths):
        canvas=Image.new('RGB',(2100,620),'white');draw=ImageDraw.Draw(canvas)
        for i,p in enumerate(paths):
            canvas.paste(Image.open(p).crop((0,90,700,670)),(i*700,40))
            rr=renderer.rows[int(np.argmin(abs(renderer.times-[1.5,35.5,55.][i])))]
            draw.text((i*700+60,8),f"t = {rr['time_s']:.1f} s | {rr['load_kg']:g} kg | {rr['angle_deg']:.1f} deg",font=font(24),fill=(23,50,71))
        canvas.save(folder/'video_preview.png')
    print('VIDEO SAVED:',path,flush=True);return path

def publication_figures(folder,target):
    """Replot the same raw data at readable report dimensions; no smoothing."""
    import matplotlib.pyplot as plt
    folder=Path(folder);target=Path(target)
    settings=json.loads((folder/'run_settings.json').read_text())
    keys=['stable_penalty','matched_no_penalty','original_penalty','fixed_PI']
    colors=['#007d9d','#d65e32','#9a6199','#b26633'];labels=['Updated penalty','Matched penalty off','Original penalty','Fixed PI']
    data={k:load_csv(folder/(k+'_data.csv')) for k in keys if (folder/(k+'_data.csv')).exists()}
    rows=data['stable_penalty'];t=np.array([r['time_s'] for r in rows]);val=lambda k:np.array([r[k] for r in rows])
    def style(axs):
        for ax in np.array(axs).flat:
            ax.tick_params(labelsize=7);ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8);ax.grid(alpha=.18)
            for event in settings['load_schedule'][1:]:ax.axvline(event['time_s'],color='gray',ls=':',lw=.6)
    def save(fig,name):
        fig.savefig(target/(name+'.pdf'),bbox_inches='tight');fig.savefig(target/(name+'.png'),dpi=200,bbox_inches='tight');plt.close(fig)
    with plt.rc_context({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7}):
        fig,axs=plt.subplots(5,1,figsize=(7,4.25),sharex=True,layout='constrained',height_ratios=[.45,1,1,1,1])
        axs[0].step(t,val('load_kg'),where='post',color='black',lw=1);axs[0].set_yticks([.5,5]);axs[0].set_ylabel('kg')
        axs[1].plot(t,val('reference_deg'),'k--',label='Target',lw=1)
        for k,c,label in zip(keys,colors,labels):
            if k not in data:continue
            rs=data[k];ts=[r['time_s'] for r in rs]
            for ax,field in zip(axs[1:],['angle_deg','error_deg','tension_N','learned_only_Nm']):ax.plot(ts,[r[field] for r in rs],color=c,label=label,lw=.8)
        for ax,label in zip(axs[1:],['Angle (deg)','Error (deg)','Force (N)','Learned (N m)']):ax.set_ylabel(label)
        axs[1].legend(loc='upper center',ncol=3,bbox_to_anchor=(.5,1.73),frameon=False,fontsize=6.5)
        axs[3].axhline(settings['tmax'],color='gray',ls=':',lw=.8);axs[-1].set_xlabel('Time (s)');style(axs);save(fig,'tracking_comparison')
        fig,axs=plt.subplots(2,1,figsize=(7,2.15),sharex=True,layout='constrained')
        for k,c,label in zip(keys[:3],colors[:3],labels[:3]):
            if k not in data:continue
            rs=data[k]
            for ax,field in zip(axs,['error_deg','tension_N']):ax.plot([r['time_s'] for r in rs],[r[field] for r in rs],color=c,label=label,lw=1)
        axs[0].set_ylabel('Error (deg)');axs[1].set_ylabel('Force (N)');axs[1].set_xlabel('Time (s)');axs[0].legend(loc='upper center',ncol=3,bbox_to_anchor=(.5,1.45),frameon=False)
        for ax in axs:ax.set_xlim(32.5,48.5)
        style(axs);save(fig,'heavy_load_zoom')
        fig,axs=plt.subplots(3,2,figsize=(7,3.7),layout='constrained');ax=axs.flat
        ax[0].plot(t,val('reference_deg'),'k--',label='Target',lw=.9);ax[0].plot(t,val('angle_deg'),label='Actual',color=colors[0],lw=1);ax[0].set_ylabel('Angle (deg)');ax[0].legend(fontsize=6,ncol=2)
        ax[1].plot(t,val('error_deg'),color=colors[1],lw=.9);ax[1].set_ylabel('Error (deg)')
        ax[2].plot(t,val('tension_N'),color='#00885d',lw=.9);ax[2].set_ylabel('Force (N)')
        for k,label in [('P_Nm','P'),('I_Nm','I'),('handcoded_PD_Nm','PD'),('learned_only_Nm','Learned'),('torque_applied_Nm','Applied')]:ax[3].plot(t,val(k),label=label,lw=.8)
        ax[3].set_ylabel('Torque (N m)');ax[3].legend(fontsize=5.5,ncol=3)
        for m,label in [('BIClong','Biceps'),('TRIlong','Triceps')]:
            ax[4].plot(t,val(m+'_fiber_length_m'),label=label,lw=.9);ax[5].plot(t,val(m+'_tendon_force_N'),label=label,lw=.9)
        ax[4].set_ylabel('Fiber length (m)');ax[5].set_ylabel('Tendon force (N)');ax[4].legend(fontsize=6,ncol=2);ax[5].legend(fontsize=6,ncol=2)
        for x in ax:x.set_xlabel('Time (s)');x.set_xlim(t[0],t[-1])
        style(axs);save(fig,'simulation_dashboard')
        diag=load_csv(folder/'learning_diagnostics.csv');fig,axs=plt.subplots(3,1,figsize=(7,2.7),sharex=True,layout='constrained')
        for x,k,c,label in zip(axs,['tracking_half_squared_rad','local_FEL_surrogate','memory_penalty'],colors,['0.5 e² (rad²)','FEL surrogate','Penalty R']):x.plot(t,[r[k] for r in diag],color=c,lw=.9);x.set_ylabel(label)
        axs[-1].set_xlabel('Time (s)');style(axs);save(fig,'learning_diagnostics')
        ck=folder/'stable_penalty/task_checkpoints.npz'
        if ck.exists():
            d=np.load(ck);fig,axs=plt.subplots(1,2,figsize=(7,2.15),layout='constrained')
            im=axs[0].imshow(d['importance_proxy'],aspect='auto',cmap='viridis');fig.colorbar(im,ax=axs[0],label='Importance')
            axs[0].set_yticks(range(len(d['masses'])),[f"Seg {int(s)}: {m:g} kg" for s,m in zip(d['segments'],d['masses'])],fontsize=7);axs[0].set_xlabel('Coefficient index')
            axs[1].plot(d['segments'],np.linalg.norm(d['weights'],axis=1),'o-',color=colors[0]);axs[1].set(xlabel='Completed segment',ylabel='Coefficient norm');axs[1].set_xticks(d['segments']);axs[1].grid(alpha=.2);save(fig,'memory_diagnostics')


FIGURE_NAMES = ['tracking_comparison','heavy_load_zoom','simulation_dashboard',
                'learning_diagnostics','memory_diagnostics','control_configuration']

def control_configuration(target):
    """Export the implemented signal paths without requiring a TeX installation."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig,ax=plt.subplots(figsize=(14,6.5),layout='constrained')
    ax.set(xlim=(0,14),ylim=(0,7));ax.axis('off')
    def box(x,y,w,text,color='#173247'):
        ax.add_patch(FancyBboxPatch((x,y),w,.85,boxstyle='round,pad=0.08',facecolor='white',edgecolor=color,lw=1.4))
        ax.text(x+w/2,y+.425,text,ha='center',va='center',fontsize=10,color=color)
    def arrow(points,label=None):
        for a,b in zip(points[:-2],points[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color='#173247',lw=1.2)
        ax.annotate('',xy=points[-1],xytext=points[-2],arrowprops=dict(arrowstyle='->',color='#173247',lw=1.2))
        if label:ax.text(*label[:2],label[2],ha='center',fontsize=9)
    box(.2,4.5,1.55,'Reference\nangle / speed')
    box(2.5,4.5,2,'Tracking errors\ne, de/dt')
    box(5.2,4.5,2,'PI + PD feedback\n68e + I + 7 de/dt')
    box(8,4.5,2,'Torque sum +\ncable conversion')
    box(11,4.5,2.6,'Arm26 + dumbbell\nnonlinear dynamics')
    for a,b in [((1.83,4.93),(2.42,4.93)),((4.58,4.93),(5.12,4.93)),((7.28,4.93),(7.92,4.93)),((10.08,4.93),(10.92,4.93))]:arrow([a,b])
    arrow([(13.68,4.93),(13.9,4.93),(13.9,3.8),(3.5,3.8),(3.5,4.42)],(8.8,3.95,'Measured angle and speed (negative feedback)'))
    ax.text(12.3,6.25,'Changing load m(t)',ha='center',fontsize=11)
    arrow([(12.3,6.05),(12.3,5.43)])
    arrow([(9,5.43),(9,5.9),(6.2,5.9),(6.2,5.43)],(7.7,6.05,'Clipping flags: anti-windup'))
    box(.3,1.8,3.3,'Features: reference, measured\nstate, and known load')
    box(4.5,1.8,2.5,'Adaptive residual\ntorque h(w)', '#007d9d')
    box(8,1.8,2.5,'Online FEL +\nEWC-inspired update','#007d9d')
    box(11.4,1.8,2.2,'Task-boundary\nmemory update')
    arrow([(3.68,2.23),(4.42,2.23)])
    arrow([(5.75,2.73),(5.75,3.25),(9,3.25),(9,4.42)],(7.7,3.4,'Learned torque'))
    arrow([(8,2.23),(7.08,2.23)],(7.55,2.45,'weights'))
    arrow([(11.32,2.23),(10.58,2.23)])
    arrow([(3.5,4.42),(3.5,3.45),(.08,3.45),(.08,.95),(9.25,.95),(9.25,1.72)],(5.9,1.08,'Tracking-error teacher s = 30e + 8 de/dt'))
    ax.text(7,.22,'Learning updates the residual coefficients; PI gains remain fixed. Cable tension is clipped to [0, Tmax].',ha='center',fontsize=10)
    ax.set_title('Adaptive (EWC-inspired) controller: implemented configuration',fontsize=15,pad=15)
    for ext in ('png','pdf'):fig.savefig(Path(target)/('control_configuration.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)

def collect_report_figures(folder,include_preview=False):
    """One authoritative directory for both the display and the report."""
    import matplotlib.pyplot as plt
    folder=Path(folder);target=folder/'figures';target.mkdir(exist_ok=True)
    publication_figures(folder,target)
    control_configuration(target)
    names=list(FIGURE_NAMES)
    if include_preview or (folder/'video_preview.png').exists():
        preview=folder/'video_preview.png'
        if not preview.exists():raise FileNotFoundError('Recorded-state video preview was not generated')
        shutil.copy2(preview,target/preview.name)
        fig,ax=plt.subplots(figsize=(14,4.2),layout='constrained')
        ax.imshow(plt.imread(preview));ax.axis('off')
        fig.savefig(target/'video_preview.pdf',bbox_inches='tight');plt.close(fig)
        names.append('video_preview')
    files=[];missing=[]
    for name in names:
        for ext in ('png','pdf'):
            path=target/(name+'.'+ext)
            ok=path.is_file() and path.stat().st_size>100
            files.append(dict(name=path.name,path=str(path.resolve()),bytes=path.stat().st_size if path.exists() else 0,ok=ok))
            if not ok:missing.append(path.name)
    manifest=dict(complete=not missing,figure_count=len(names),file_count=len(files),files=files,missing=missing)
    (target/'figure_manifest.json').write_text(json.dumps(manifest,indent=2))
    if missing:raise RuntimeError('Missing figure exports: '+', '.join(missing))
    print(f'FIGURES VERIFIED: {len(names)} figures, {len(files)} PNG/PDF files in {target.resolve()}',flush=True)
    return manifest

def show_all_figures(folder):
    """Open every exported report figure, including diagnostics closed during export."""
    import matplotlib.pyplot as plt
    folder=Path(folder);target=folder/'figures'
    manifest=json.loads((target/'figure_manifest.json').read_text())
    if not manifest['complete']:raise RuntimeError('Cannot display an incomplete figure export')
    if 'agg' == plt.get_backend().lower():
        print('DISPLAY UNAVAILABLE: Matplotlib uses the noninteractive Agg backend. All files are saved. Run without --no-show using a desktop Python environment.',flush=True)
        return False
    plt.close('all')
    pngs=[target/x['name'] for x in manifest['files'] if x['name'].endswith('.png')]
    for i,path in enumerate(pngs,1):
        fig,ax=plt.subplots(figsize=(12,7),layout='constrained')
        fig.canvas.manager.set_window_title(f'Arm26 report figure {i}/{len(pngs)}: {path.stem}')
        ax.imshow(plt.imread(path));ax.axis('off')
        ax.set_title(f'{i}/{len(pngs)}  '+path.stem.replace('_',' ').title(),fontsize=13)
    print(f'DISPLAY: opening all {len(pngs)} report figures. Close the figure windows to finish.',flush=True)
    plt.show(block=True)
    return True


def build_overleaf_report(folder):
    """Create a self-contained editable report from the current run's figures/data."""
    # Prefix the parent before joining: the directory can be shorter than 260
    # characters while a copied figure inside report_overleaf/figures is longer.
    folder=filesystem_path(folder);settings=json.loads((folder/'run_settings.json').read_text())
    with (folder/'segment_metrics.csv').open() as f:metrics=list(csv.DictReader(f))
    labels={'stable_penalty':'Adaptive (EWC-inspired)','matched_no_penalty':'Matched penalty off','original_penalty':'Original penalty learner','fixed_PI':'Fixed PI baseline'}
    grouped={key:[r for r in metrics if r['controller']==key] for key in labels}
    overall={r['controller']:r['overall_rmse_deg'] for r in settings['summaries']}
    def metric(key,seg):
        row=next((x for x in grouped.get(key,[]) if int(x['segment'])==seg),None)
        return float(row['rmse_deg']) if row else float('nan')
    table=[]
    for key,label in labels.items():
        if key not in overall:continue
        vals=[metric(key,i) for i in range(4)]+[overall[key]]
        table.append(label+' & '+' & '.join(f'{v:.3f}' for v in vals)+r'\\')
    total_samples=len(load_csv(folder/'stable_penalty_data.csv'))
    rows=load_csv(folder/'stable_penalty_data.csv')
    saturated=sum(r['saturated'] for r in rows)
    sat='The updated controller does not clip its command in this recorded run.' if saturated==0 else f'The updated command clips in {100*saturated/len(rows):.2f} percent of samples.'
    rep={
        '@@RUN_ID@@':settings['run_id'].replace('_',r'\_'),
        '@@OBSERVATION@@':'known mass modulation' if settings.get('known_mass_input',True) else 'a constant duplicate block; true mass is hidden',
        '@@LAMBDA@@':f"{settings['ewc_lambda']:g}", '@@LR@@':f"{settings['nn_lr']:g}", '@@DURATION@@':f"{settings['duration']:g}",
        '@@DT@@':f"{settings['dt']:g}", '@@CAP@@':f"{settings['tmax']:g}", '@@HEAVY@@':f"{metric('stable_penalty',2):.3f}",
        '@@OVERALL@@':f"{overall['stable_penalty']:.3f}", '@@OLD_HEAVY@@':f"{metric('original_penalty',2):.3f}",
        '@@NAIVE_HEAVY@@':f"{metric('matched_no_penalty',2):.3f}", '@@NAIVE_OVERALL@@':f"{overall['matched_no_penalty']:.3f}",
        '@@RETURN@@':f"{metric('stable_penalty',3):.3f}", '@@NAIVE_RETURN@@':f"{metric('matched_no_penalty',3):.3f}",
        '@@TABLE@@':'\n'.join(table), '@@SATURATION@@':sat,
        '@@COMPLETION@@':f'All cases completed; {total_samples:,} samples per case.' if settings['completed'] else 'At least one case is incomplete; inspect trial_status.json before interpreting aggregate scores.'}
    templates=filesystem_path(ROOT)
    text=(templates/'report_template.tex').read_text(encoding='utf-8')
    for key,value in rep.items():text=text.replace(key,value)
    report=folder/'report_overleaf';figs=report/'figures';figs.mkdir(parents=True,exist_ok=True)
    for src in (folder/'figures').iterdir():
        if src.suffix in ('.png','.pdf'):shutil.copy2(src,figs/src.name)
    shutil.copy2(templates/'control_block_diagram.tex',report/'control_block_diagram.tex')
    (report/'main.tex').write_text(text,encoding='utf-8')
    (report/'README.txt').write_text('Upload this folder or its ZIP to Overleaf. Select main.tex and pdfLaTeX. All required figures are included. The video is separate.\n')
    import zipfile
    with zipfile.ZipFile(folder/'report_overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
        for src in report.rglob('*'):
            if src.is_file() and (src.parent==figs or src.name in ['main.tex','control_block_diagram.tex','README.txt']):z.write(src,src.relative_to(report).as_posix())
    return report


def print_saved_files(folder,status):
    print('\n========== SAVED OUTPUTS ==========',flush=True)
    print('SAVE FOLDER: '+str(folder.resolve()),flush=True)
    for p in sorted((folder/'figures').glob('*')):
        if p.suffix in ('.png','.pdf'):print('FIGURE: '+str(p.resolve()),flush=True)
    if status.get('video'):print('MOVIE: '+status['video'],flush=True)
    if status.get('report'):print('OVERLEAF: '+status['report'],flush=True)
    for key,value in status.get('errors',{}).items():print('EXPORT ERROR ('+key+'): '+value,flush=True)
    print('===================================\n',flush=True)


def export_all(folder,geometry=None,make_video=True,fps=10):
    folder=Path(folder).expanduser().resolve()
    status={'folder':str(folder),'figures':False,'video':None,'report':None,'errors':{}}
    print('EXPORT STARTED: '+str(folder),flush=True)
    try:
        diagnostic_figures(folder);status['figures']=True
    except Exception as exc:
        status['errors']['figures']=type(exc).__name__+': '+str(exc)
        (folder/'figure_export_error.txt').write_text(traceback.format_exc())
    if make_video:
        try:
            meta_path=folder/'video_metadata.json';cached=json.loads(meta_path.read_text()) if meta_path.exists() else {}
            existing=folder/cached.get('video','__missing__')
            end=json.loads((folder/'run_settings.json').read_text())['duration']
            if existing.is_file() and existing.stat().st_size>1024 and cached.get('fps')==fps and abs(cached.get('rendered_through_s',-1)-end)<=1/fps:
                video=existing;print('MOVIE: complete existing replay retained.',flush=True)
            else:video=export_video(folder,geometry,fps)
            status['video']=str(video.resolve())
        except Exception as exc:
            status['errors']['video']=type(exc).__name__+': '+str(exc)
            (folder/'video_error.txt').write_text(traceback.format_exc())
    else:status['video_skipped']=True
    try:
        manifest=collect_report_figures(folder,include_preview=bool(status['video']))
        status['figures']=manifest['complete'];status['figure_count']=manifest['figure_count']
        status['figures_folder']=str(folder/'figures')
    except Exception as exc:
        status['figures']=False
        status['errors']['figure_collection']=type(exc).__name__+': '+str(exc)
        (folder/'figure_collection_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
    if status['figures'] and 'figure_collection' not in status['errors']:
        try:
            build_overleaf_report(folder);status['report']=str(folder/'report_overleaf.zip')
        except Exception as exc:
            # A report failure must not relabel verified PNG/PDF files as failed.
            status['errors']['report']=type(exc).__name__+': '+str(exc)
            (folder/'report_export_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            print('REPORT ERROR: see '+str(folder/'report_export_error.txt'),flush=True)
    (folder/'export_status.json').write_text(json.dumps(status,indent=2))
    print_saved_files(folder,status)
    if status['errors']:raise RuntimeError('Export incomplete; see export_status.json. Simulation data are preserved.')
    return status


def latest_results(root=ROOT):
    candidates=[]
    for pattern in ['stable_tracking_*/run_settings.json','simulation_outputs/stable_tracking_*/run_settings.json',
                    '../*/stable_tracking_*/run_settings.json','../*/simulation_outputs/stable_tracking_*/run_settings.json']:
        for p in root.glob(pattern):
            if (p.parent/'stable_penalty_data.csv').is_file():candidates.append(p.parent.resolve())
    if not candidates:raise FileNotFoundError('No completed stable-tracking run found nearby. Pass its full folder path to SAVE_EXISTING_RUN.py.')
    return max(set(candidates),key=lambda p:(p/'run_settings.json').stat().st_mtime)


def recover_exports(folder=None,geometry=None,make_video=True,fps=10,show=True):
    if not show:
        import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from types import SimpleNamespace
    from run_arm26_stable_tracking import plot_results
    folder=latest_results() if folder is None or str(folder).lower()=='latest' else Path(folder).expanduser().resolve()
    (ROOT/'LATEST_RESULTS_PATH.txt').write_text(str(folder.resolve())+'\n')
    print('RECOVER EXPORTS: '+str(folder.resolve()),flush=True)
    settings=json.loads((folder/'run_settings.json').read_text())
    trials={k:load_csv(folder/(k+'_data.csv')) for k in ['stable_penalty','matched_no_penalty','original_penalty','fixed_PI'] if (folder/(k+'_data.csv')).is_file()}
    plot_results(plt,trials,SimpleNamespace(**settings),settings['load_schedule'],folder)
    status=export_all(folder,geometry,make_video,fps)
    if show:show_all_figures(folder)
    else:plt.close('all')
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path,nargs='?');p.add_argument('--geometry');p.add_argument('--fps',type=int,default=10);p.add_argument('--no-video',action='store_true');p.add_argument('--max-seconds',type=float);p.add_argument('--no-show',action='store_true',help='Save every figure without opening figure windows.')
    args=p.parse_args()
    if args.max_seconds is not None:
        folder=args.results or latest_results();export_video(folder,args.geometry,args.fps,args.max_seconds)
    else:recover_exports(args.results,args.geometry,not args.no_video,args.fps,show=not args.no_show)
if __name__=='__main__':main()

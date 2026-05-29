import sys, math, yaml, gmsh

def load_cfg(path="config.txt"):
    with open(path) as f: c=yaml.safe_load(f)
    for m in c["materials"].values():
        if "E" in m: m["E"]=float(m["E"])
        if "c" in m: m["c"]=float(m["c"])
    return c

cfg=load_cfg("config.txt")
g=cfg["geometry"]
L   = float(g["tusk_length_total"])      # 1.80 m (config). 3D mesh is 2.564 m.
Rob,Rot = g["outer_radius_base"], g["outer_radius_tip"]
Rib,Rit = g["inner_radius_base"], g["inner_radius_tip"]
tcb,tct = g["cementum_thickness_base"], g["cementum_thickness_tip"]

# wavelength-driven sizing
F0=25e3; N_LAMBDA=12; N_THRU=4
cs=math.sqrt(float(cfg["materials"]["dentine"]["E"])/(2*1900*(1+0.30)))
lam_min=0.85*cs/F0                  # ~A0-ish slow guided mode
h_lambda=lam_min/N_LAMBDA
wall_b=(Rob-tcb)-Rib; wall_t=(Rot-tct)-Rit
h_base=min(h_lambda, wall_b/N_THRU); h_tip=min(h_lambda, wall_t/N_THRU)
print(f"cs~{cs:.0f} lam_min~{1e3*lam_min:.1f}mm  h_base~{1e3*h_base:.2f}mm h_tip~{1e3*h_tip:.2f}mm")

gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
gmsh.model.add("tusk_axi")
occ=gmsh.model.occ
def rin(z):  return Rib +(Rit -Rib )*z/L
def rdo(z):  return (Rob-tcb)+((Rot-tct)-(Rob-tcb))*z/L   # dentine outer = cementum inner
def rout(z): return Rob +(Rot -Rob )*z/L
def quad(r0f,r1f):  # trapezoid (r0(z)->r1(z)) over z in [0,L], in (x=r,y=z)
    p=[occ.addPoint(r0f(0),0,0),occ.addPoint(r1f(0),0,0),
       occ.addPoint(r1f(L),L,0),occ.addPoint(r0f(L),L,0)]
    l=[occ.addLine(p[i],p[(i+1)%4]) for i in range(4)]
    return occ.addPlaneSurface([occ.addCurveLoop(l)])
s_pulp=quad(lambda z:0.0, rin); s_dent=quad(rin,rdo); s_cem=quad(rdo,rout)
occ.fragment([(2,s_pulp),(2,s_dent),(2,s_cem)],[])   # conforming interfaces
occ.synchronize()

def near(a,b,t=1e-7): return abs(a-b)<t
T=cfg["mesh"]["domain_tags"]; B=cfg["mesh"]["boundary_tags"]
# tag surfaces by centroid radius
for dim,tag in gmsh.model.getEntities(2):
    x,y,_=gmsh.model.occ.getCenterOfMass(2,tag); zr=y
    if   x<rin(zr):  gmsh.model.addPhysicalGroup(2,[tag],T["pulp"],    "pulp")
    elif x<rdo(zr):  gmsh.model.addPhysicalGroup(2,[tag],T["dentine"], "dentine")
    else:            gmsh.model.addPhysicalGroup(2,[tag],T["cementum"],"cementum")
# tag boundary curves
axis=[];base=[];tip=[];outer=[];inner=[]
for dim,tag in gmsh.model.getEntities(1):
    x,y,_=gmsh.model.occ.getCenterOfMass(1,tag)
    bb=gmsh.model.getBoundingBox(1,tag); dz=bb[4]-bb[1]; dr=bb[3]-bb[0]
    if near(x,0.0): axis.append(tag)
    elif near(y,0.0): base.append(tag)
    elif near(y,L):   tip.append(tag)
    elif dr>dz:       pass
    elif near(x,rout(y),5e-4):           # outer cementum surface
        outer.append(tag)
    elif near(x,rin(y),5e-4): inner.append(tag)
gmsh.model.addPhysicalGroup(1,axis ,50,"axis")
gmsh.model.addPhysicalGroup(1,base ,B["jaw_fixed"],"jaw_fixed")
gmsh.model.addPhysicalGroup(1,tip  ,B["tusk_tip"], "tusk_tip")
gmsh.model.addPhysicalGroup(1,outer,B["outer_surface"],"outer_surface")
gmsh.model.addPhysicalGroup(1,inner,B["inner_surface"],"inner_surface")

# graded size field: h grows linearly base->? no: refine where wall thin (tip)
f=gmsh.model.mesh.field
f.add("MathEval",1)
f.setString(1,"F", f"{h_base}+({h_tip}-{h_base})*y/{L}")
f.setAsBackgroundMesh(1)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints",0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary",0)
gmsh.option.setNumber("Mesh.Algorithm",6)
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(int(cfg["mesh"]["polynomial_order"]))
gmsh.write("tusk_axi.msh")

# report
ntypes={2:"tri3",9:"tri6"}
for dim,tg in gmsh.model.getPhysicalGroups(2):
    nm=gmsh.model.getPhysicalName(dim,tg);n=0
    for e in gmsh.model.getEntitiesForPhysicalGroup(dim,tg):
        et,ets,_=gmsh.model.mesh.getElements(dim,e)
        for t,a in zip(et,ets): n+=len(a); ty=ntypes.get(t,t)
    print(f"  VOL {nm:9s} tag{tg}: {n} {ty}")
for dim,tg in gmsh.model.getPhysicalGroups(1):
    nm=gmsh.model.getPhysicalName(dim,tg);n=0
    for e in gmsh.model.getEntitiesForPhysicalGroup(dim,tg):
        et,ets,_=gmsh.model.mesh.getElements(dim,e)
        for t,a in zip(et,ets): n+=len(a)
    print(f"  BND {nm:18s} tag{tg}: {n}")
nn,_,_=gmsh.model.mesh.getNodes(); print("total nodes:",len(nn))
gmsh.finalize()

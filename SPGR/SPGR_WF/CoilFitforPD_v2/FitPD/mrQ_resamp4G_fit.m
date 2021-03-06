 function [opt]=mrQ_resamp4G_fit(opt,outMm,interp,T1reg)
 %This function resample the M0f (4D)  T1 and brain mask (3D) images to
 %the the outMm resulotion and save those file location to the opt so they
 %will be used for Gain Fit. interp define the interpolation method
 %according to the SPM standart use in mrAnatResliceSpm.m
 %
 %[opt]=mrQ_resamp4G_fit(opt,outMm,interp)
 % use mrQ_UpSampG4PD_fit.m
 %M0file='/biac2/wandell2/data/WH/008_AM/Qmr/20111020_1294_32ch_1mm3/20111020_1294/SPGR_2/Align_0.9375_0.9375_1/AligncombineCoilsM0.nii.gz'

 if notDefined('outMm')
     outMm=[2 2 2];
 end
 
 if notDefined('interp')
     interp = 1;

 end
 
 opt.Resamp=1;

 
 
 %% multi coil M0

M0=readFileNifti(opt.M0file);
sz=size(M0.data);
bb = mrAnatXformCoords(M0.qto_xyz, [1 1 1; sz(1:3)]);

for ii=1:sz(4)
[M0UnderSamp(:,:,:,ii), M0UnderSamp_Xform] = mrAnatResliceSpm(double(M0.data(:,:,:,ii)), inv(M0.qto_xyz), bb, outMm, interp, 0);

end

clear M0 

filename=fullfile(opt.outDir,['M0resmp_' num2str(outMm(1)) '_' num2str(outMm(2)) '_' num2str(outMm(3)) '.nii.gz']);
dtiWriteNiftiWrapper(single(M0UnderSamp),M0UnderSamp_Xform,filename);

opt.M0file_Org=opt.M0file;
opt.M0file=filename;
 clear  M0UnderSamp M0UnderSamp_Xform filename

%% Brain mask
BM=readFileNifti(opt.BMfile);


sz=size(BM.data);
%bb = mrAnatXformCoords(BM.qto_xyz, [1 1 1; sz(1:3)]);
[BMUnderSamp, BMUnderSamp_Xform] = mrAnatResliceSpm(double(BM.data(:,:,:)), inv(BM.qto_xyz), bb, outMm, interp, 0);


%% T1
if opt.T1reg
T1=readFileNifti(opt.T1file);


[T1UnderSamp, T1UnderSamp_Xform] = mrAnatResliceSpm(double(T1.data(:,:,:)), inv(T1.qto_xyz), bb, outMm, interp, 0);
%% clean and save T1 and BM
%% clean:
% this values are probably out of the origunal brian mask and are result of
% resample process
T1UnderSamp((T1UnderSamp<0.4))=0; % brian tisuue
T1UnderSamp((T1UnderSamp>5))=5; % too high T1
BMUnderSamp(T1UnderSamp==0)=0; % not brain by T1 values
end

BMUnderSamp(BMUnderSamp<1)=0; %not sure this is brain any more

B=ones(3,3,3) ;% a 3X3x3 mask
C=convn(BMUnderSamp,B,'same'); %to be sure lets only keep the voxel  that are neen brain voxel
BMUnderSamp(C<sum(B(:)))=0;


%% save
%BM
filename=fullfile(opt.outDir,['BMresmp_' num2str(outMm(1)) '_' num2str(outMm(2)) '_' num2str(outMm(3)) '.nii.gz']);
dtiWriteNiftiWrapper(single(BMUnderSamp),BMUnderSamp_Xform,filename);
opt.BMfile_Org=opt.BMfile;
opt.BMfile=filename;
%clear BM BMUnderSamp BMUnderSamp_Xform filename

if opt.T1reg
%T1
filename=fullfile(opt.outDir,['T1resmp_' num2str(outMm(1)) '_' num2str(outMm(2)) '_' num2str(outMm(3)) '.nii.gz']);
dtiWriteNiftiWrapper(single(T1UnderSamp),T1UnderSamp_Xform,filename);


opt.T1file_Org=opt.T1file;
opt.T1file=filename;
end
%clear T1 T1UnderSamp T1UnderSamp_Xform filename








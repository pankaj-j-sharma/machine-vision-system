import { Component, EventEmitter, HostListener, OnInit, Output } from '@angular/core';


export interface uploadFile {
  no: number
  file : File 
  name:string
  path:string
  updated: number
};

@Component({
  selector: 'app-file-upload',
  templateUrl: './file-upload.component.html',
  styleUrls: ['./file-upload.component.scss']
})


export class FileUploadComponent implements OnInit {

  dragAreaClass: string = "";
  loadedImgUrl: string = "";
  fileList : uploadFile[] = [];

  @Output() onFilesDropped = new EventEmitter<any>();

  constructor() { }

  ngOnInit(): void {
    this.dragAreaClass = "dragarea";
  }

  @HostListener("dragover", ["$event"]) onDragOver(event: any) {
    this.dragAreaClass = "droparea";
    event.preventDefault();
  }
  @HostListener("dragenter", ["$event"]) onDragEnter(event: any) {
    this.dragAreaClass = "droparea";
    event.preventDefault();
  }
  @HostListener("dragend", ["$event"]) onDragEnd(event: any) {
    //this.dragAreaClass = "dragarea";
    event.preventDefault();
  }
  @HostListener("dragleave", ["$event"]) onDragLeave(event: any) {
    //this.dragAreaClass = "dragarea";
    event.preventDefault();
  }
  @HostListener("drop", ["$event"]) onDrop(event: any) {
    //this.dragAreaClass = "dragarea";
    event.preventDefault();
    event.stopPropagation();

    if (event.dataTransfer.files) {
      let files: FileList = event.dataTransfer.files;
      console.log('files upload component uploaded',files.length,'files');
      this.saveFiles(files);
      this.onFilesDropped.emit(this.fileList);
    }
  }  

  saveFiles(files: FileList) {
      
    this.fileList = [];
    for(let i=0; i < files.length; i++){

      this.fileList.push({'no':i,'file':files[i],'name':files[i].name,'path':'','updated':files[i].lastModified});        
      var reader = new FileReader();

      // Closure to capture the file information.
      reader.onload = ((theFile) => {
          return (e:any) => {
            this.fileList.filter(f=>f.name==theFile.name)[0].path=e.target.result;              
          };
      })(files[i]);

      // Read in the image file as a data URL.
      reader.readAsDataURL(files[i]);        
    }
}

onFileChange(event:any){
  console.log('event fired ->',event)
  this.saveFiles(event.target.files)
}

removeUpload(){
  this.fileList=[];
}

}

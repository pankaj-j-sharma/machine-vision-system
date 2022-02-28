import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';

var $:any;

@Component({
  selector: 'app-evaluation',
  templateUrl: './evaluation.component.html',
  styleUrls: ['./evaluation.component.scss']
})
export class EvaluationComponent implements OnInit {

  @ViewChild('iframeContainer')
  iframeContainer!: ElementRef;

  constructor() { }

  ngOnInit(): void {
  }

  ngAfterViewInit(): void {
    this.injectIframe();
  }

  injectIframe(): void {

    let BASE_URL = window.location.hostname=="localhost" ? "http://localhost:6006/" : "http://"+window.location.hostname+"/tensorAPI/";
    
    const container = this.iframeContainer.nativeElement;
    const iframe = document.createElement('iframe');
    iframe.setAttribute('width', '100%');
    iframe.setAttribute('id', 'myiFrame');
    // iframe.setAttribute('src', 'http://localhost:6006/');
    iframe.setAttribute('src', BASE_URL);
    iframe.setAttribute('height', '100%');
    iframe.setAttribute('frameBorder', '0');
    iframe.addEventListener('load', this.iframeOnLoadtwo);
    container.appendChild(iframe);
  
  }

  iframeOnLoadtwo(): void {
    console.log('iframe loaded...');
  }
}

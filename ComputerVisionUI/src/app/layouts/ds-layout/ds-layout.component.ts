import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-ds-layout',
  templateUrl: './ds-layout.component.html',
  styleUrls: ['./ds-layout.component.scss']
})
export class DsLayoutComponent implements OnInit {

  constructor(private router : Router) { }

  ngOnInit(): void {
    console.log('routers avalable ',this.router.config)
  }
}

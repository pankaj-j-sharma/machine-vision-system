import { HttpEvent, HttpHandler, HttpInterceptor, HttpRequest, HttpResponse } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';
import { UUID } from 'angular2-uuid';
import { appAlertItems, appMsgItems, homePageAppItems, navSideBarItem } from './appdata';

@Injectable({
  providedIn: 'root'
})
export class ApiInterceptorService implements HttpInterceptor {

  constructor() { }
  intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    console.log('intercepted ->',req.url);
    console.log('data',req.body);
    // if (req.url.endsWith('/login')){
    // return of(new HttpResponse({ status: 200, body: {token:UUID.UUID()} }));
    // }
    // if (req.url.endsWith('/gethomeapps')){
    //   return of(new HttpResponse({ status: 200, body: {homeApps: homePageAppItems} }));
    // }
    // if (req.url.endsWith('/getappalerts')){
    //   return of(new HttpResponse({ status: 200, body: {alertItems: appAlertItems} }));
    // }

    if (req.url.endsWith('/getmenuitem')){
      let navSideBarItemArr =  navSideBarItem.filter(n=>n.roles.includes(req.body));
      navSideBarItemArr.forEach(i=> i.url=(i.url=="/home" ||i.url.startsWith('processtracking') ? i.url: i.url.replace("/","/"+req.body+"-")));
      return of(new HttpResponse({ status: 200, body: {results:navSideBarItemArr}}));
    }

    if (req.url.endsWith('/getusermsgs')){
      return of(new HttpResponse({ status: 200, body: {allMsgs: appMsgItems} }));
    }
    
    
    return next.handle(req);
  }
}
